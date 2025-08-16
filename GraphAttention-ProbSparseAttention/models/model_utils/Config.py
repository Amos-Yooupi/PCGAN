import json


class Config(object):
    def __init__(self, config_path: str):
        super().__init__()
        for key, value in self.load_config(config_path).items():
            setattr(self, key, value)

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
