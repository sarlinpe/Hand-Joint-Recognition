import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import experiment
from jointrecog.settings import EXPER_PATH


def augment_data(data):
    images = np.stack([np.rot90(data['image'], k) for k in range(4)])
    dummy_kp = np.stack([data['keypoints'] for _ in range(4)])
    return {'image': images, 'keypoints': dummy_kp}


def merge_kp_after_augment(kp):
    shape = (128, 128)
    kp = [k[0] for k in np.split(kp, 4, axis=0)]
    straight_kp = []
    for i, k in enumerate(kp):
        if i == 0:
            straight_kp.append(k)
        elif i == 1:
            straight_kp.append([0, shape[0]] + [1, -1]*k[:, ::-1])
        elif i == 2:
            straight_kp.append(shape - k)
        elif i == 3:
            straight_kp.append([shape[1], 0] + [-1, 1]*k[:, ::-1])
    return np.mean(straight_kp, axis=0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--export_name', type=str, default=None)
    parser.add_argument('--augment', dest='augment', action='store_true')
    args = parser.parse_args()

    experiment_name = args.experiment_name
    export_name = args.export_name if args.export_name else experiment_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_file = Path(EXPER_PATH, '{}.csv'.format(export_name))
    checkpoint_dir = Path(EXPER_PATH, experiment_name)

    with experiment._init_graph(config, with_dataset=True) as (net, dataset):
        net.load(str(checkpoint_dir), flexible_restore=False)
        test_set = dataset.get_test_set()
        predictions = []

        pbar = tqdm(total=None)
        while True:
            try:
                data = next(test_set)
            except dataset.end_set:
                break

            if args.augment:
                data = augment_data(data)
            kp = net.predict(data, keys='keypoints', batch=args.augment)
            if args.augment:
                kp = merge_kp_after_augment(kp)

            kp = np.flip(kp, axis=-1)
            predictions += list(kp.reshape(1, 2*21))
            pbar.update(1)

    coloumns = []
    for i in range(21):
        coloumns += ['Joint %d x' % i]
        coloumns += ['Joint %d y' % i]

    final_output = pd.DataFrame(predictions, columns=coloumns)
    final_output.index.name = 'Id'
    final_output.to_csv(output_file)
