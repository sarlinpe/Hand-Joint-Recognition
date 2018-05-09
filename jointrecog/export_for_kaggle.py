import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import experiment
from jointrecog.settings import EXPER_PATH

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment_name', type=str)
    args = parser.parse_args()

    experiment_name = args.experiment_name
    export_name = experiment_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_file = Path(EXPER_PATH, 'outputs/{}.csv'.format(export_name))
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
            kp = net.predict(data, keys='keypoints')
            predictions += list(kp.reshape(1, 2*21))
            pbar.update(1)

    print(predictions)

    coloumns = []
    for i in range(21):
        coloumns += ['Joint %d x' % i]
        coloumns += ['Joint %d y' % i]

    final_output = pd.DataFrame(predictions, columns=coloumns)
    final_output.index.name = 'Id'
    final_output.to_csv(output_file)
