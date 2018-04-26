import tensorflow as tf
from tensorflow import layers as tfl

from .base_model import BaseModel, Mode
from .backbones import resnet_v2 as resnet

class JointRecognition(BaseModel):
    input_spec = {
            'image': {'shape': [None, 128, 128, 3], 'type': tf.float32},
            'score_map' : {'shape' : [None, 128, 128, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {'data_format': 'channels_first'}

    def jointrecog_model(self, inputs, mode, **config):
        x = inputs['image']
        if config['data_format'] == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])

        with slim.arg_scope(resnet.resnet_arg_scope()):
            is_training = Mode.TRAIN
            with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=is_training):
                _, encoder = resnet.resnet_v2_50(image,
                                             is_training=is_training,
                                             global_pool=False,
                                             scope='resnet_v2_50')
        feature_map = encoder['resnet_v2_50/block3']

        with slim.arg_scope(resnet.resnet_arg_scope()):
            is_training = config['train_backbone'] and (mode == Mode.TRAIN)
            with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=is_training):
                x = slim.conv2d(feature_map,
                        config['num_keypoints'], kernel_size = [1, 1],
                        rate = 8.0, activation_fn = None, scope = 'convDilateDilated')

                size = tf.shape(inputs)[1:3]
                # resize the output logits to match the labels dimensions
                # net = tf.image.resize_nearest_neighbor(net, size)
                x = tf.image.resize_bilinear(x, size)

        if mode == Mode.TRAIN:
            return {'score_map': x}
        else:
            return {'score_map': x}

    def _loss(self, outputs, inputs, **config):
        # TODO
        with tf.name_scope('loss'):
            loss = 0
        return loss

    def _metrics(self, outputs, inputs, **config):
        #TODO
        metrics = {}
        with tf.name_scope('metrics'):
            metrics['l2dist'] = 0
        return metrics
