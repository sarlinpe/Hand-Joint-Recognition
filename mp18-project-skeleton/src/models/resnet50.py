"""MnistNet architecture."""
from typing import Dict

import tensorflow as tf
from tensorflow.contrib import slim

from core import BaseDataSource, BaseModel
from .backbones import resnet_v2 as resnet

class Resnet50(BaseModel):
    """MnistNet architecture as used in [Zhang et al. CVPR'15]."""


    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['img']
        y = input_tensors['kp_2D']

        with tf.variable_scope('conv'):
            with slim.arg_scope(resnet.resnet_arg_scope()):
                _, encoder = resnet.resnet_v2_50(x
                                    is_training=True,
                                    global_pool=False,
                                    scope='resnet_v2_50')
                x = encoder['resnet_v2_50/block3']

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
