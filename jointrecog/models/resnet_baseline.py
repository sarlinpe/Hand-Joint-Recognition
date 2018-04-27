lmport tensorflow as tf
from tensorflow import layers as tfl

from .base_model import BaseModel, Mode
from .backbones import resnet_v2 as resnet

NUM_JOINTS = 21

class JointRecognition(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 3], 'type': tf.float32},
            'scoremap' : {'shape' : [None, None, None], 'type': tf.float32},
            'keypoints' : {'shape' : [None, NUM_JOINTS, 2], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {'data_format': 'channels_first'}

    def jointrecog_model(self, inputs, mode, **config):
        x = inputs['image']
        shape = tf.shape(inputs['image'])[1:3]
        if config['data_format'] == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])

        # Resnet Backbone
        with slim.arg_scope(resnet.resnet_arg_scope()):
            is_training = Mode.TRAIN
            with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=is_training):
                _, encoder = resnet.resnet_v2_50(image,
                                             is_training=is_training,
                                             global_pool=False,
                                             scope='resnet_v2_50')
        feature_map = encoder['resnet_v2_50/block3']

        # Upsampling of feature map to input size
        with slim.arg_scope(resnet.resnet_arg_scope()):
            is_training = config['train_backbone'] and (mode == Mode.TRAIN)
            with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=is_training):
                # Resize feature map to match label dimensions
                x = tf.image.resize_bilinear(x, shape)
                # Apply convolution to obtain one tensor slice per keypoint
                x = slim.conv2d(
                        x, NUM_JOINTS, kernel_size = config['output_kernel'],
                        rate = 1.0, activation_fn = None, scope = 'convOut')

        with slim.name_scope('prediction'):
            x_grid = tf.to_float(tf.expand_dims(tf.expand_dims(
                tf.tile(tf.expand_dims(tf.range(shape[0]), 1), [1, shape[1]]), -1), 0))
            y_grid = tf.to_float(tf.expand_dims(tf.expand_dims(
                tf.tile(tf.expand_dims(tf.range(shape[1]), 0), [shape[0], 1]), -1), 0))

            with slim.name_scope('expectation'):
                # Predict joint locations from expectation values of scoremaps
                # Calculate expectation as weighted avg of grid * scoremap
                expectation_x = tf.reduce_sum(x_grid * outputs['scoremap'], axis=[1, 2]))
                expectation_y = tf.reduce_sum(y_grid * outputs['scoremap'], axis=[1, 2]))
                expectations_x /= tf.reduce_sum(outputs['scoremap'], axis=[1,2])
                expectations_y /= tf.reduce_sum(outputs['scoremap'], axis=[1,2])
                expectations = tf.stack([expectations_x, expectations_y], 2)

            with slim.name_scope('maxscore'):
                # Predict joint locations from pixels with maximum score
                mask = tf.equal(outputs['score'], tf.reduce_max(outputs['score'],
                    axis= [1, 2], keepdims=True)
                max_scores = tf.reduce_max(tf.to_float(mask) * x_gridi, axis=[1, 2i, axis=[1, 2]])

        return {'scoremap': x,
                'expectations' : expectations,
                'maxscores' : max_scores}

    def _loss(self, outputs, inputs, **config):
        with tf.name_scope('loss'):
            d = inputs['scoremap'] - outputs['scoremap']
            loss = tf.reduce_mean(tf.square(d))
        return loss

    def _metrics(self, outputs, inputs, **config):
        metrics = {}
        with tf.name_scope('metrics'):
            shape = tf.shape(outputs['scoremap'])
            x_grid = tf.to_float(tf.expand_dims(tf.expand_dims(
                tf.tile(tf.expand_dims(tf.range(shape[0]), 1), [1, shape[1]]), -1), 0))
            y_grid = tf.to_float(tf.expand_dims(tf.expand_dims(
                tf.tile(tf.expand_dims(tf.range(shape[1]), 0), [shape[0], 1]), -1), 0))
            # Calculate expectation as weighted avg of grid * scoremap
            expectation_x = tf.reduce_sum(x_grid * outputs['scoremap'], axis=[1, 2]))
            expectation_y = tf.reduce_sum(y_grid * outputs['scoremap'], axis=[1, 2]))
            expectations_x /= tf.reduce_sum(outputs['scoremap'], axis=[1,2])
            expectations_y /= tf.reduce_sum(outputs['scoremap'], axis=[1,2])

            expectations = tf.stack([expectations_x, expectations_y], 2)
            d = inputs['keypoints'] - expectations
            metrics['l2dist'] = tf.reduce_mean(tf.square(d))
        return metrics
