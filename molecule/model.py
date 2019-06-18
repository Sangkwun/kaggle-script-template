import xgboost as xgb
import lightgbm as lgb
import catboost as cat

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge



RANDOM_SEED = 0

"""
    Target model
    1) Lgb
    2) Xgb
    3) RF(Random forest)
    4) Ridge
    5) 

"""

class BaseModel(object):
    def __init__(self, params, **kawrgs):
        self.param = params
        pass

    def train(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
        #return predict
        
    def get_params(self):
        return self.params


class LgbModel(BaseModel):
    # https://lightgbm.readthedocs.io/en/latest/

    def __init__(self, params):
        super(LgbModel, self).__init__(params)
        self.model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)

    def train(self):
        pass


class XgbModel(BaseModel):
    # https://xgboost.readthedocs.io/en/latest/
    def __init__(self, params):
        super(LgbModel, self).__init__(params)



class CatModel(BaseModel):
    def __init__(self, params):
        super(LgbModel, self).__init__(params)


def build_model(cfg):
    model = LgbModel(cfg)
    return model