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
            'num_threads': 10,
            'preprocessing': {
                'resize': [128, 128],
                'bbox_margin': 8,
                'bbox_symmetric': False,
            },
            'augmentation': {
                'bbox_distort': False,
                'random_rotation': False,
                'random_brightness': False,
                'random_saturation': False,
            },
            'disk_radius': None,
            'scoremap_variance': None
    }

    def create_split_file(self, filepath, config):
        tf.logging.info('Creating validation split file.')
        with h5py.File(Path(DATA_PATH, self.train_filename), 'r') as hf:
            indices = list(range(len(hf['train']['img'])))
        random.Random(config['shuffle_seed']).shuffle(indices)
        split = {'validation': indices[:config['validation_size']],
                 'training': indices[config['validation_size']:]}
        with open(str(filepath), 'w') as f:
            json.dump(split, f)

    def _init_dataset(self, **config):
        split_filepath = Path(DATA_PATH, self.split_filename)
        if not split_filepath.exists():
            self.create_split_file(split_filepath, config)

        with open(str(split_filepath), 'r') as f:
            split = json.load(f)
        with h5py.File(Path(DATA_PATH, self.test_filename), 'r') as hf:
            split['test'] = list(range(len(hf['test']['img'])))
        return split

    def _get_data(self, split, split_name, **config):
        d = tf.data.Dataset.from_tensor_slices(split[split_name])
        filename = self.test_filename if split_name == 'test' else self.train_filename
        data_file = h5py.File(Path(DATA_PATH, filename), 'r')

        # TODO: use a generator if too slow:
        # https://stackoverflow.com/questions/48309631/tensorflow-tf-data-dataset-reading-large-hdf5-files
        def _read_element(idx, entry, has_label):
            img = data_file[entry]['img'][idx]
            img = img.astype(np.float32)
            if has_label:
                kp = data_file[entry]['kp_2D'][idx].astype(np.float32)
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

        def _base_bbox(img, kp):
            margin = config['preprocessing']['bbox_margin']
            shape = tf.shape(img)[:2]
            # Find extreme keypoints
            kp_min = tf.to_int32(tf.floor(tf.reduce_min(kp, axis=0)))
            kp_max = tf.to_int32(tf.ceil(tf.reduce_max(kp, axis=0)))
            # Add margin to get bound box corners
            bb_min = tf.maximum(kp_min - margin, 0)
            bb_max = tf.minimum(kp_max + margin, shape-1)
            # Expand the edge of smaller size, symmetrically or not
            if config['preprocessing']['bbox_symmetric']:
                bwidth = tf.reduce_max(bb_max - bb_min) / 2
                center = (bb_max + bb_min) / 2
                bb_min = tf.to_int32(center - bwidth)
                bb_max = tf.to_int32(center + bwidth)
            else:
                bb_max = tf.to_int32(bb_min + tf.reduce_max(bb_max - bb_min))
            # Correct if out of bounds
            correction_min = tf.minimum(bb_min, 0)
            correction_max = tf.maximum(bb_max - (shape-1), 0)
            bb_max -= correction_min + correction_max
            bb_min -= correction_min + correction_max
            return img, kp, tf.stack([bb_min, bb_max])

        def _augment_bbox(img, kp, bbox):
            shape = tf.to_float(tf.shape(img)[:2])
            bb_min, bb_max = tf.unstack(tf.to_float(bbox), num=2)
            # Random translation
            margins = tf.reduce_min(tf.stack([
                tf.reduce_min(kp, axis=0) - bb_min, bb_max - tf.reduce_max(kp, axis=0),
                bb_min, shape - 1 - bb_max]), axis=0)
            offset = tf.round(tf.stack([
                tf.truncated_normal([], stddev=margins[i]/2) for i in [0, 1]]))
            bb_min += offset
            bb_max += offset
            # Random scaling
            margin = tf.reduce_min(tf.stack([
                tf.reduce_min(kp, axis=0) - bb_min, bb_max - tf.reduce_max(kp, axis=0),
                bb_min, shape - 1 - bb_max]))
            offset = tf.round(tf.truncated_normal([], stddev=margin/2))
            bb_min -= offset
            bb_max += offset
            return img, kp, tf.to_int32(tf.stack([bb_min, bb_max]))

        def _crop_bbox(img, kp, bbox):
            bb_min, bb_max = tf.unstack(bbox, num=2)
            img = img[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], ...]
            kp -= tf.expand_dims(tf.to_float(bb_min), 0)
            return img, kp

        def _resize(img, kp):
            kp /= tf.to_float(tf.shape(img)[:2])
            kp *= tf.constant(config['preprocessing']['resize'], dtype=tf.float32)
            img = tf.image.resize_images(img, config['preprocessing']['resize'],
                                         method=tf.image.ResizeMethod.BILINEAR)
            return img, kp

        def _random_rotation(img, kp):
            shape = tf.to_float(tf.shape(img)[:2] - 1)
            rot = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k=rot)
            kp = tf.case({
                tf.equal(rot, 0): lambda: kp,
                tf.equal(rot, 1): lambda: [shape[0], 0] + [-1, 1]*tf.reverse(kp, [-1]),
                tf.equal(rot, 2): lambda: shape - kp,
                tf.equal(rot, 3): lambda: [0, shape[1]] + [1, -1]*tf.reverse(kp, [-1])},
                exclusive=True)
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
            if config['scoremap_variance'] is not None:
                data.update({'scoremap': tf.exp(
                        -distance/(2*tf.constant(float(config['scoremap_variance']))))})
            if config['disk_radius'] is not None:
                data.update({'disks': tf.less_equal(
                        distance, tf.to_float(tf.square(config['disk_radius'])))})
            return data

        if split_name == 'test':
            d = d.map(
                    lambda idx: tf.py_func(
                        lambda idx: _read_element(idx, 'test', False),
                        [idx], tf.float32), num_parallel_calls=config['num_threads'])
            d = d.map(_preprocess_image, num_parallel_calls=config['num_threads'])
            d = d.map(lambda i: (i, tf.zeros([self.num_keypoints, 2])))  # dummy kp
        else:
            d = d.map(
                    lambda idx: tuple(tf.py_func(
                        lambda idx: _read_element(idx, 'train', True),
                        [idx], [tf.float32, tf.float32])),
                    num_parallel_calls=config['num_threads'])
            d = d.map(
                    lambda img, kp: (_preprocess_image(img), _preprocess_keypoints(kp)))

            # Bbox processing
            d = d.map(_base_bbox, num_parallel_calls=config['num_threads'])
            if config['cache_in_memory']:
                d = d.cache()
            if split_name == 'training' and config['augmentation']['bbox_distort']:
                d = d.map(_augment_bbox, num_parallel_calls=config['num_threads'])
            d = d.map(_crop_bbox, num_parallel_calls=config['num_threads'])
            d = d.map(_resize, num_parallel_calls=config['num_threads'])

            if split_name == 'training':
                if config['augmentation']['random_rotation']:
                    d = d.map(_random_rotation, num_parallel_calls=config['num_threads'])
                if config['augmentation']['random_brightness']:
                    d = d.map(lambda img, kp: (tf.clip_by_value(
                        tf.image.random_brightness(img, 25), 0, 255), kp))
                if config['augmentation']['random_saturation']:
                    d = d.map(lambda img, kp: (tf.clip_by_value(
                        tf.image.random_saturation(img, .5, 1.5), 0, 255), kp))

        d = d.map(lambda image, kp: {'image': image, 'keypoints': kp})
        d = d.map(_add_label_maps, num_parallel_calls=config['num_threads'])

        return d
