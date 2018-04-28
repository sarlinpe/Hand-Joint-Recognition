import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v2 as resnet

NUM_JOINTS = 21

class ResnetBaseline(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 3], 'type': tf.float32},
            'scoremap' : {'shape': [None, None, None, 1], 'type': tf.float32},
            'keypoints' : {'shape': [None, NUM_JOINTS, 2], 'type': tf.float32},
            'disks' : {'shape': [None, None, None, 1], 'type': tf.float32}
    }

    default_config = { 'data': {'data_format': 'channels_last'}, 'model': {'output_kernel': [3, 3]}}

    def _model(self, inputs, mode, **config):
        image = inputs['image']
        print(inputs.keys())
        if config['data']['data_format'] == 'channels_first':
            image = tf.transpose(image, [0, 3, 1, 2])

        with tf.name_scope('prediction'):

            is_training = (mode == Mode.TRAIN)
            shape = tf.shape(image)[1:3]

            with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=is_training):
                with slim.arg_scope(resnet.resnet_arg_scope()):
                    # Resnet Backbone
                    _, encoder = resnet.resnet_v2_50(image,
                                             is_training=is_training,
                                             global_pool=False,
                                             scope='resnet_v2_50')
                    feature_map = encoder['resnetbaseline/resnet_v2_50/block3']

                # Upsampling of feature map to input size
                image = tf.image.resize_bilinear(image, [shape[0], shape[1]])
                # Apply convolution to obtain one tensor slice per keypoint
                scores = slim.conv2d(
                        image, NUM_JOINTS, kernel_size = config['model']['output_kernel'],
                        activation_fn = None, scope = 'convOut')

            x_grid = tf.to_float(tf.expand_dims(tf.expand_dims(
                tf.tile(tf.expand_dims(tf.range(shape[0]), 1), [1, shape[1]]), -1), 0))
            y_grid = tf.to_float(tf.expand_dims(tf.expand_dims(
                tf.tile(tf.expand_dims(tf.range(shape[1]), 0), [shape[0], 1]), -1), 0))

            with tf.name_scope('expectation'):
                # Predict joint locations from expectation values of scoremaps
                # Calculate expectation as weighted avg of grid * scoremap
                expectations_x = tf.reduce_sum(x_grid * scores, axis=[1, 2])
                expectations_y = tf.reduce_sum(y_grid * scores, axis=[1, 2])
                expectations_x /= tf.reduce_sum(scores, axis=[1,2])
                expectations_y /= tf.reduce_sum(scores, axis=[1,2])
                expectations = tf.stack([expectations_x, expectations_y], 2)

            with tf.name_scope('maxscore'):
                # Predict joint locations from pixels with maximum score
                mask = tf.equal(scores,
                    tf.reduce_max(scores, axis=[1, 2], keepdims=True))
                max_scores = tf.reduce_max(tf.to_float(mask) * x_grid, axis=[1, 2])

        return {'scoremap': scores,
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
            shape = tf.shape(outputs['scoremap'])[1:3]
            x_grid = tf.to_float(tf.expand_dims(tf.expand_dims(
                tf.tile(tf.expand_dims(tf.range(shape[0]), 1), [1, shape[1]]), -1), 0))
            y_grid = tf.to_float(tf.expand_dims(tf.expand_dims(
                tf.tile(tf.expand_dims(tf.range(shape[1]), 0), [shape[0], 1]), -1), 0))
            # Calculate expectation as weighted avg of grid * scoremap
            expectations_x = tf.reduce_sum(x_grid * outputs['scoremap'], axis=[1, 2])
            expectations_y = tf.reduce_sum(y_grid * outputs['scoremap'], axis=[1, 2])
            expectations_x /= tf.reduce_sum(outputs['scoremap'], axis=[1,2])
            expectations_y /= tf.reduce_sum(outputs['scoremap'], axis=[1,2])

            expectations = tf.stack([expectations_x, expectations_y], 2)
            d = inputs['keypoints'] - expectations
            metrics['l2dist'] = tf.reduce_mean(tf.square(d))
        return metrics
