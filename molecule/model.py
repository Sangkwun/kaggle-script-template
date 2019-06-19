import xgboost as xgb
import lightgbm as lgb
#import catboost as cat


from .utils import group_mean_log_mae
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn import metrics

metrics_dict = {
    'mae': {
        'lgb_metric_name': 'mae',
        'catboost_metric_name': 'MAE',
        'scoring_function': metrics.mean_absolute_error
    },
    'group_mae': {
        'lgb_metric_name': 'mae',
        'catboost_metric_name': 'MAE',
        'scoring_function': group_mean_log_mae
    },
    'mse': {
        'lgb_metric_name': 'mse',
        'catboost_metric_name': 'MSE',
        'scoring_function': metrics.mean_squared_error
    }
}

RANDOM_SEED = 0

"""
    Target model
    1) Lgb
    2) Xgb
    3) RF(Random forest)
    4) Ridge
    5) 

    output
    - oof df: index + prediction
    - config
    - score
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

    def __init__(self, params, metric, verbose=1000, early_stopping_rounds=200):
        super(LgbModel, self).__init__(params)

        self.model = lgb.LGBMRegressor(**params)
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.metric = metrics_dict[metric]['lgb_metric_name']
        self.evaluate = metrics_dict[metric]['scoring_function']

    def train(self, train_x, train_y, valid_x, valid_y):
        self.model.fit(
            train_x, 
            train_y, 
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric=self.metric,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds
        )
    
    def predict(self, x_test):
        y_pred = self.model.predict(x_test, num_iteration=self.model.best_iteration_)
        return y_pred


class XgbModel(BaseModel):
    # https://xgboost.readthedocs.io/en/latest/
    def __init__(self, params):
        super(LgbModel, self).__init__(params)



class CatModel(BaseModel):
    def __init__(self, params):
        super(LgbModel, self).__init__(params)


def build_model(cfg):
    if cfg.type == 'lgb':
        model = LgbModel(cfg.params, cfg.metric, cfg.verbose, cfg.early_stopping_rounds)
    elif cfg.type == 'xgb':
        model = LgbModel(cfg.params, cfg.metric, cfg.verbose, cfg.early_stopping_rounds)
    else:
        raise NotImplementedError
    
    return model