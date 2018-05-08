#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf
import os
from settings import DATA_PATH, EXPER_PATH

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Train a 2D joint estimation model.')
    parser.add_argument('exper_name', type=str)
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    exper_name = args.exper_name
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

        # Declare some parameters
        batch_size = 32

        # Define model
        from datasources import HDF5Source
        from models import ExampleNet
        model = ExampleNet(
            # Tensorflow session
            # Note: The same session must be used for the model and the data sources.
            session,

            # Set experiment output directory
            output_dir=os.path.join(EXPER_PATH, exper_name),

            # The learning schedule describes in which order which part of the network should be
            # trained and with which learning rate.
            #
            # A standard network would have one entry (dict) in this argument where all model
            # parameters are optimized. To do this, you must specify which variables must be
            # optimized and this is done by specifying which prefixes to look for.
            # The prefixes are defined by using `tf.variable_scope`.
            #
            # The loss terms which can be specified depends on model specifications, specifically
            # the `loss_terms` output of `BaseModel::build_model`.
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {
                        'kp_2D_mse': ['conv', 'fc'],
                    },
                    'metrics': ['kp_2D_mse'],
                    'learning_rate': 1e-4,
                },
            ],

            test_losses_or_metrics=['kp_2D_mse'],

            # Data sources for training and testing.
            train_data={
                'real': HDF5Source(
                    session,
                    batch_size,
                    hdf_path=os.path.join(DATA_PATH,'training.h5'),
                    keys_to_use=['train'],
                    min_after_dequeue=2000,
                ),
            },
            # If you want to validate your model, split the training set into
            # training and validation and uncomment this line
            # test_data={
            #     'real': HDF5Source(
            #         session,
            #         batch_size,
            #         hdf_path=os.path.join(DATA_PATH,'validation.h5'),
            #         keys_to_use=['test'],
            #         testing=True,
            #     ),
            # },
        )

        # Train this model for a set number of epochs
        model.train(
            num_epochs=50,
        )

        # Test for Kaggle submission
        model.evaluate_for_kaggle(
            HDF5Source(
                session,
                batch_size,
                hdf_path='../datasets/testing.h5',
                keys_to_use=['test'],
                testing=True,
            )
        )
