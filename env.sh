env_dir=./env

if [ ! -d "$env_dir" ]; then
    python -m pip install --user virtualenv
    python -m virtualenv \
        --system-site-packages \
        --python="$(which python3)" \
        "$env_dir"
fi

source "$env_dir/bin/activate"
