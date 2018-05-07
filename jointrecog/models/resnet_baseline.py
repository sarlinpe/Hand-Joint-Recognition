import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v2 as resnet

NUM_KEYPOINTS = 21


class ResnetBaseline(BaseModel):
    input_spec = {
        'image': {'shape': [None, None, None, 3], 'type': tf.float32},
        'scoremap' : {'shape': [None, None, None, NUM_KEYPOINTS], 'type': tf.float32},
        'keypoints' : {'shape': [None, NUM_KEYPOINTS, 2], 'type': tf.float32},
        'disks' : {'shape': [None, None, None, NUM_KEYPOINTS], 'type': tf.float32}
    }

    default_config = {
            'output_kernel': 3
    }

    def _model(self, inputs, mode, **config):
        image = inputs['image']

        with tf.name_scope('prediction'):
            is_training = (mode == Mode.TRAIN)
            input_shape = tf.shape(image)[1:3]

            with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=is_training):
                with slim.arg_scope(resnet.resnet_arg_scope()):
                    # Resnet Backbone
                    _, encoder = resnet.resnet_v2_50(
                            image, is_training=is_training, global_pool=False,
                            scope='resnet_v2_50')
                    features = encoder['resnet_v2_50/block1']

                # Upsampling of feature map to input size
                features = tf.image.resize_bilinear(features, input_shape)
                # Apply convolution to obtain one  per keypoint
                scores = slim.conv2d(
                        features, NUM_KEYPOINTS,
                        kernel_size=config['output_kernel'],
                        activation_fn=None,
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
                expectations_x = tf.reduce_sum(x_grid * scores, axis=[1, 2])
                expectations_y = tf.reduce_sum(y_grid * scores, axis=[1, 2])
                expectations_x /= tf.reduce_sum(scores, axis=[1, 2])
                expectations_y /= tf.reduce_sum(scores, axis=[1, 2])
                expectations = tf.stack([expectations_x, expectations_y], 2)

            with tf.name_scope('maxscore'):
                # Predict joint locations from pixels with maximum score
                mask = tf.equal(scores, tf.reduce_max(scores, axis=[1, 2],
                                keepdims=True))
                max_scores = tf.reduce_max(tf.to_float(mask) * x_grid, axis=[1, 2])

        return {'scoremap': scores,
                'expectations': expectations,
                'maxscores': max_scores}

    def _loss(self, outputs, inputs, **config):
        with tf.name_scope('loss'):
            d = inputs['scoremap'] - outputs['scoremap']
            loss = tf.reduce_mean(tf.square(d))
        return loss

    def _metrics(self, outputs, inputs, **config):
        metrics = {}
        with tf.name_scope('metrics'):
            diff = tf.square(inputs['keypoints'] - outputs['expectations'])
            metrics['l2dist'] = tf.reduce_mean(tf.reduce_sum(diff, axis=[1, 2]))
        return metrics
