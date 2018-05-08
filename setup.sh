read -p "Path of the directory where datasets are stored and read: " dir
echo "DATA_PATH = '$dir'" >> ./jointrecog/settings.py
echo "DATA_PATH = '$dir'" >> ./mp18-project-skeleton/src/settings.py

read -p "Path of the directory where experiments data (logs, checkpoints, configs) are written: " dir
echo "EXPER_PATH = '$dir'" >> ./jointrecog/settings.py
echo "EXPER_PATH = '$dir'" >> ./mp18-project-skeleton/src/settings.py
