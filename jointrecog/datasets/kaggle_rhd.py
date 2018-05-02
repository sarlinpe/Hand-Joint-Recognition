import h5py
import tensorflow as tf
import json
import random
import numpy as np
from pathlib import Path

from .base_dataset import BaseDataset
from jointrecog.settings import DATA_PATH


class KaggleRhd(BaseDataset):
    split_filename = 'split.json'
    train_filename = 'training.h5'
    test_filename = 'testing.h5'
    num_keypoints = 21

    default_config = {
            'shuffle_seed': 0,
            'validation_size': 5000,
            'cache_in_memory': False,
            'preprocessing': {
                'resize': [128, 128],
                'bbox_margin': 8,
            },
            'disk_radius': 7,
            'scoremap_variance': 25
    }

    def create_split_file(self, filepath, config):
        tf.logging.info('Creating validation split file.')
        with h5py.File(Path(DATA_PATH, self.train_filename), 'r') as hf:
            indices = list(range(len(hf['train']['img'])))
        random.Random(config['shuffle_seed']).shuffle(indices)
        split = {'validation': indices[:config['validation_size']],
                 'training': indices[config['validation_size']:]}
        with open(filepath, 'w') as f:
            json.dump(split, f)

    def _init_dataset(self, **config):
        split_filepath = Path(DATA_PATH, self.split_filename)
        if not split_filepath.exists():
            self.create_split_file(split_filepath, config)

        with open(split_filepath, 'r') as f:
            split = json.load(f)
        with h5py.File(Path(DATA_PATH, self.test_filename), 'r') as hf:
            split['test'] = list(range(len(hf['test']['img'])))
        return split

    def _get_data(self, split, split_name, **config):
        d = tf.data.Dataset.from_tensor_slices(split[split_name])

        # TODO: use a generator if too slow:
        # https://stackoverflow.com/questions/48309631/tensorflow-tf-data-dataset-reading-large-hdf5-files
        def _read_element(idx, entry, filepath, has_label):
            with h5py.File(filepath, 'r') as hf:
                img = hf[entry]['img'][idx]
                img = img.astype(np.float32)
                if has_label:
                    kp = hf[entry]['kp_2D'][idx].astype(np.float32)
                    return img, kp
                else:
                    return img

        def _preprocess_image(img):
            # BGR to RGB and NHW to HWN
            img = tf.reverse(tf.transpose(img, [1, 2, 0]), [-1])
            img.set_shape([None, None, 3])
            return img

        def _preprocess_keypoints(kp):
            kp.set_shape([None, 2])
            return tf.reverse(kp, [-1])

        def _crop_bbox(img, kp):
            margin = config['preprocessing']['bbox_margin']
            shape = tf.shape(img)[:2]
            # Find extreme keypoints
            kp_min = tf.to_int32(tf.floor(tf.reduce_min(kp, axis=0)))
            kp_max = tf.to_int32(tf.ceil(tf.reduce_max(kp, axis=0)))
            # Add margin to get bound box corners
            bb_min = tf.maximum(kp_min - margin, 0)
            bb_max = tf.minimum(kp_max + margin, shape-1)
            # Expand the edge of smaller size symmetrically
            bwidth = tf.reduce_max(bb_max - bb_min) / 2
            center = (bb_max + bb_min) / 2
            bb_min = tf.to_int32(center - bwidth)
            bb_max = tf.to_int32(center + bwidth)
            # Correct if out of bounds
            correction_min = tf.minimum(bb_min, 0)
            correction_max = tf.maximum(bb_max - (shape-1), 0)
            bb_max -= correction_min + correction_max
            bb_min -= correction_min + correction_max
            # Perform the cropping
            img = img[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], ...]
            kp -= tf.expand_dims(tf.to_float(bb_min), 0)
            return img, kp

        def _resize(img, kp):
            kp /= tf.to_float(tf.shape(img)[:2])
            kp *= tf.constant(config['preprocessing']['resize'], dtype=tf.float32)
            img = tf.image.resize_images(img, config['preprocessing']['resize'],
                                         method=tf.image.ResizeMethod.BILINEAR)
            return img, kp

        def _add_label_maps(data):
            shape = tf.shape(data['image'])[:2]
            kp = data['keypoints']
            x_grid = tf.to_float(
                    tf.tile(tf.expand_dims(tf.range(shape[0]), 1), [1, shape[1]]))
            y_grid = tf.to_float(
                    tf.tile(tf.expand_dims(tf.range(shape[1]), 0), [shape[0], 1]))
            x_diff = tf.expand_dims(x_grid, -1) - tf.reshape(kp[:, 0], [1, 1, -1])
            y_diff = tf.expand_dims(y_grid, -1) - tf.reshape(kp[:, 1], [1, 1, -1])
            distance = tf.square(x_diff) + tf.square(y_diff)
            scoremap = tf.exp(
                    -distance/(2*tf.constant(float(config['scoremap_variance']))))
            disks = tf.less_equal(
                    distance, tf.to_float(tf.square(config['disk_radius'])))
            return {**data, **{'scoremap': scoremap, 'disks': disks}}

        if split_name == 'test':
            d = d.map(
                    lambda idx: tf.py_func(
                        lambda idx: _read_element(
                            idx, 'test', Path(DATA_PATH, self.test_filename), False),
                        [idx], tf.float32))
            d = d.map(_preprocess_image)
            d = d.map(lambda i: (i, tf.zeros([self.num_keypoints, 2])))  # dummy kp
        else:
            d = d.map(
                    lambda idx: tuple(tf.py_func(
                        lambda idx: _read_element(
                            idx, 'train', Path(DATA_PATH, self.train_filename), True),
                        [idx], [tf.float32, tf.float32])))
            d = d.map(
                    lambda img, kp: (_preprocess_image(img), _preprocess_keypoints(kp)))
            d = d.map(_crop_bbox)
            d = d.map(_resize)

        d = d.map(lambda image, kp: {'image': image, 'keypoints': kp})
        d = d.map(_add_label_maps)

        if config['cache_in_memory']:
            tf.logging.info('Caching dataset, fist access will take some time.')
            d = d.cache()

        return d
