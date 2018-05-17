import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v2 as resnet
from .utils import kaggle_mse

NUM_KEYPOINTS = 21


class ResnetDisks(BaseModel):
    input_spec = {
        'image': {'shape': [None, None, None, 3], 'type': tf.float32},
        'keypoints': {'shape': [None, NUM_KEYPOINTS, 2], 'type': tf.float32},
        'disks': {'shape': [None, None, None, NUM_KEYPOINTS], 'type': tf.bool}
    }

    default_config = {
            'output_kernel': 3
    }

    def _model(self, inputs, mode, **config):
        is_training = (mode == Mode.TRAIN)
        x = inputs['image']
        input_shape = tf.shape(x)[1:3]

        with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=is_training):
            with slim.arg_scope(resnet.resnet_arg_scope()):
                # Resnet Backbone
                _, encoder = resnet.resnet_v2_152(
                                        x,
                                        is_training=is_training,
                                        global_pool=False,
                                        input_pool=False,
                                        scope='resnet_v2_152')
                x = encoder['resnet_v2_152/block2']

            # Upsampling of feature map to input size
            x = tf.image.resize_bilinear(x, input_shape)
            # Apply convolution to obtain one  per keypoint
            if is_training:
                output_activation = None
            else:
                output_activation = tf.nn.sigmoid
            logits = slim.conv2d(
                    x, NUM_KEYPOINTS,
                    kernel_size=config['output_kernel'],
                    activation_fn=output_activation,
                    scope='convOut')

        x_grid, y_grid = tf.meshgrid(
            tf.to_float(tf.range(input_shape[0])),
            tf.to_float(tf.range(input_shape[1])),
            indexing='ij')
        x_grid = tf.expand_dims(tf.expand_dims(x_grid, 0), -1)
        y_grid = tf.expand_dims(tf.expand_dims(y_grid, 0), -1)

        with tf.name_scope('expectation'):
            # Predict joint locations from expectation values of scoremaps
            # Calculate expectation as weighted avg of grid * scoremap
            expectations_x = tf.reduce_sum(x_grid * logits, axis=[1, 2])
            expectations_y = tf.reduce_sum(y_grid * logits, axis=[1, 2])
            expectations_x /= tf.reduce_sum(logits, axis=[1, 2])
            expectations_y /= tf.reduce_sum(logits, axis=[1, 2])
            expectations = tf.stack([expectations_x, expectations_y], 2)

        with tf.name_scope('max_probability'):
            # Predict joint locations from pixels with maximum score
            mask = tf.equal(logits, tf.reduce_max(logits, axis=[1, 2],
                            keepdims=True))
            p_max_x = tf.reduce_max(tf.to_float(mask) * x_grid, axis=[1, 2])
            p_max_y = tf.reduce_max(tf.to_float(mask) * y_grid, axis=[1, 2])
            p_max = tf.stack([p_max_x, p_max_y], axis=-1)

        return {'logits': logits,
                'expectations': expectations,
                'p_max': p_max}

    def _loss(self, outputs, inputs, **config):
        labels = tf.cast(inputs['disks'], dtype=tf.int64)
        logits = outputs['logits']
        with tf.name_scope('loss'):
            loss = tf.losses.sigmoid_cross_entropy(labels, logits)
            print(loss.get_shape().as_list())
            loss = tf.reduce_mean(loss)
        return loss

    def _metrics(self, outputs, inputs, **config):
        with tf.name_scope('metrics'):
            metrics = {'mse_exp': kaggle_mse(outputs['expectations'],
                                             inputs['keypoints']),
                       'mse_maxp': kaggle_mse(outputs['p_max'],
                                              inputs['keypoints'])}
        return metrics
