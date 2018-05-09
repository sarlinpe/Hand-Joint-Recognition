import tensorflow as tf

from .base_model import BaseModel
from .utils import kaggle_mse

NUM_KEYPOINTS = 21


class MnistFc(BaseModel):
    """MnistNet architecture as used in [Zhang et al. CVPR'15]."""
    input_spec = {
        'image': {'shape': [None, 128, 128, 3], 'type': tf.float32},
        'keypoints': {'shape': [None, NUM_KEYPOINTS, 2], 'type': tf.float32}
    }

    def _model(self, inputs, mode, **config):
        """Build model."""
        x = inputs['image']
        x.set_shape([None, 128, 128, 3])
        with tf.variable_scope('conv'):
            # The MnistNet architecture is formed of 2 convs and 2 fcs
            num_filters = (32, 32, 32)
            for j, n in enumerate(num_filters):
                x = tf.layers.conv2d(x, filters=n, kernel_size=3, strides=1,
                                     padding='same', data_format='channels_last')

                x = tf.nn.relu(x)

                x = tf.layers.max_pooling2d(x, pool_size=3, strides=2,
                                            padding='same', data_format='channels_last')
        with tf.variable_scope('fc'):
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=2*NUM_KEYPOINTS, name='out')
            x = tf.reshape(x, (-1, 21, 2))

        return {'keypoints': x}

    def _loss(self, outputs, inputs, **config):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.squared_difference(
                                        inputs['keypoints'], outputs['keypoints']))
        return loss

    def _metrics(self, outputs, inputs, **config):
        with tf.name_scope('metrics'):
            metrics = {'mse': kaggle_mse(outputs['keypoints'], inputs['keypoints'])}
        return metrics
