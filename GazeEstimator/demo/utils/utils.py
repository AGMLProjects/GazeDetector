import pathlib

import yaml


def load_demo_configs() -> dict:
    config_path = pathlib.Path('../config/demo/demo.yaml')
    with open(config_path, 'r') as conf:
        data = yaml.load(conf, Loader=yaml.FullLoader)
        return data
