import argparse
import pandas as pd
import numpy as np

from pathlib import Path

from .model import build_model
from .utils import get_config
from .dataset import load_dataset, feature_engineering, split_fold

def arg_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'predict', 'split_fold'])
    arg('--model_configs', nargs='+') # path for model config
    arg('--n_fold', type=int, default= 5)

    args = parser.parse_args()
    return args

def split(n_fold):
    train_df, test_df = load_dataset()
    train_df = split_fold(train_df, n_fold)
    train_df.to_csv('./train.csv', index=False)
    test_df.to_csv('./test.csv', index=False)

def main():
    args = arg_parser()
    if args.mode == 'split_fold':
        return split(args.n_fold)

    folds = pd.read_csv('train.csv', dtype={"fold": int})
    test_df = pd.read_csv('test.csv')

    if args.mode == 'train':
        train(args.model_configs, folds, test_df)
    elif args.mode == 'predict':
        raise NotImplementedError
    elif args.mode == 'valid':
        raise NotImplementedError

def train(model_configs, folds, test_df):
    config_root = Path('configs')
    models = list()

    for path in model_configs:
        config_path = config_root / path
        cfg = get_config(config_path)
        model = build_model(cfg)
        models.append(model)

    oof = np.zeros(len(folds))
    prediction = np.zeros(len(test_df))
    scores = []

    for fold in folds['fold'].unique():
        train_fold = folds[folds['fold'] != fold]
        valid_fold = folds[folds['fold'] == fold]
        valid_index = valid_fold['id']

        x_train = train_fold.drop(['id', 'molecule_name', 'scalar_coupling_constant', 'fold'], axis=1)
        y_train = train_fold['scalar_coupling_constant']
        x_valid = valid_fold.drop(['id', 'molecule_name', 'scalar_coupling_constant', 'fold'], axis=1)
        y_valid = valid_fold['scalar_coupling_constant']
        x_test = test_df.drop(['id', 'molecule_name'], axis=1)

        for model in models:
            result = model.train(x_train, y_train, x_valid, y_valid)
            y_pred_val = model.predict(x_valid)
            y_pred = model.predict(x_test)
            score = model.evaluate(y_valid, y_pred_val)
            oof[valid_index] = y_pred_val.reshape(-1,)
            prediction += y_pred
    
    prediction /= len(folds['fold'].unique())*len(models)
    sub = pd.read_csv('../input/sample_submission.csv')
    sub['scalar_coupling_constant'] = prediction
    sub.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()