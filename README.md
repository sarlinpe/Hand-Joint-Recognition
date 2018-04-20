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
