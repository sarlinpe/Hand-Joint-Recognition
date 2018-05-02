# Hand-Joint-Recognition
Recognizing individual 2D joint locations in images

### Installation
```
. env.sh
make install
```

### Usage
The launcher script is `experiment.py` with the command arguments `python experiment.py <command> <config_path> <experiment_name>` where:
  - `<command>` is one of `train`, `evaluate`, `predict`
  - `<config_path>` is the path to the configuration file
  - `<experiment_name>` is the name of the experiment under which the output files will be saved

Example:
```
cd jointrecog/
python experiment.py train configs/example.yaml test_mnit
```

### Load Pre-Trained Weights
In order to load pre-trained weights create a directory weights in your data directory
```
mkdir <DATA_DIR>/weights
cd <DATA_DIR>/weights
wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
tar -xzf resnet_v2_50_2017_04_14.tar.gz
```
