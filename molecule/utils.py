import json


class Config(object):
    def __init__(self, config_path):
        with open(config_path, encoding='utf-8') as f:
            cfg = json.loads(f.read())
        for key, value in cfg.items():
            setattr(self, key, value)
        # type and attribute check in here

def get_config(cfg):
    cfg = Config(cfg)
    return cfg