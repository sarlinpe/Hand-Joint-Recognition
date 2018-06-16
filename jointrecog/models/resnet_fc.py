import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v2 as resnet
from .utils import kaggle_mse

NUM_KEYPOINTS = 21


class ResnetFc(BaseModel):
    input_spec = {
        'image': {'shape': [None, 128, 128, 3], 'type': tf.float32},
        'keypoints': {'shape': [None, NUM_KEYPOINTS, 2], 'type': tf.float32}
    }

    def _model(self, inputs, mode, **config):
        is_training = (mode == Mode.TRAIN)
        x = inputs['image']
        x.set_shape([None, 128, 128, 3])

        with slim.arg_scope(resnet.resnet_arg_scope()):
                _, encoder = resnet.resnet_v2_152(
                                            x,
                                            is_training=is_training,
                                            global_pool=False,
                                            scope='resnet_v2_152')
                x = encoder['resnet_v2_152/block4']
        with tf.variable_scope('fc'):
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=2*NUM_KEYPOINTS, name='out')
            x = tf.reshape(x, (-1, NUM_KEYPOINTS, 2))

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
