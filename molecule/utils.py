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

def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()