"""MnistNet architecture."""
from typing import Dict

import tensorflow as tf

from core import BaseDataSource, BaseModel

class ExampleNet(BaseModel):
    """MnistNet architecture as used in [Zhang et al. CVPR'15]."""


    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['img']
        y = input_tensors['kp_2D']

        with tf.variable_scope('conv'):
            # The MnistNet architecture is formed of 2 convs and 2 fcs
            num_filters = (32, 32, 32)
            for j, n in enumerate(num_filters):
                # Make sure to use channels_first or NCHW format for image data.
                # cuDNN is optimized for and expects NCHW format whereas Tensorflow default
                # is NHWC (as it is faster on the CPU with SIMD.
                x = tf.layers.conv2d(x, filters=n, kernel_size=3, strides=1,
                                     padding='same', data_format='channels_first')

                x = tf.nn.relu(x)

                x = tf.layers.max_pooling2d(x, pool_size=3, strides=2,
                                            padding='same', data_format='channels_first')

        with tf.variable_scope('fc'):
            # Flatten the 50 feature maps down to one vector
            x = tf.contrib.layers.flatten(x)

            # Directly regress two polar angles for gaze direction
            x = tf.layers.dense(x, units=42, name='out')

            x = tf.reshape(x, (-1, 21, 2))

        # Define outputs
        loss_terms = {  # To optimize
            'kp_2D_mse': tf.reduce_mean(tf.squared_difference(x, y)),
        }
        return {'kp_2D': x}, loss_terms, {}
