import tensorflow as tf


def kaggle_mse(predicted_keypoints, groundtruth_keypoints):
    diff = tf.square(predicted_keypoints - groundtruth_keypoints)
    return tf.reduce_mean(tf.reduce_sum(diff, axis=-1))
