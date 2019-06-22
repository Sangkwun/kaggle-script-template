import argparse
import pandas as pd
import numpy as np

from pathlib import Path

from .model import build_model
from .utils import get_config
from .dataset import load_dataset, split_fold


"""
    Todo:
    1) train과 predict를 split할지 말지?
    2) 안한다면 stacking ensemble구현 방식
        - csv로 쓰고 이를 다시 학습에 사용하는 방식으로 stacking구현
    3) model config위치 validation
    4) feature set 다양하게할 방법?
    
"""
def arg_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['run', 'split_fold', 'ensemble'])
    arg('--train_csv', type=str, default='train.csv')
    arg('--test_csv', type=str, default='test.csv')

    # for run
    arg('--run_name', default='stage1', type=str)
    arg('--model_configs', nargs='+')

    # for split_fold
    arg('--n_fold', type=int, default= 5)

    # for ensemble
    arg('--ensemble_name', default='ensemble1', type=str)
    arg('--csv_paths', nargs='+')

    args = parser.parse_args()
    return args

def split(args):
    train_df, test_df = load_dataset()
    train_df = split_fold(train_df, args.n_fold)
    train_df.to_csv(args.train_csv, index=False)
    test_df.to_csv(args.test_csv, index=False)

def main():
    args = arg_parser()
    if args.mode == 'split_fold':
        return split(args)

    if args.mode == 'run':
        run(args)
    elif args.mode == 'ensemble':
        ensemble(args)
    else:
        raise NotImplementedError

def run(args):
    folds = pd.read_csv(args.train_csv, dtype={"fold": int})[:30]
    test_df = pd.read_csv(args.test_csv)[:20]

    config_root = Path('configs')
    models = list()

    for path in args.model_configs:
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
            oof[valid_index] += y_pred_val.reshape(-1,)
            prediction += y_pred

    oof /= len(models)
    prediction /= len(folds['fold'].unique())*len(models)
    
    folds['scalar_coupling_constant'] = oof
    folds.to_csv('{}.csv'.format(args.run_name), index=False)

    sub = pd.read_csv('../input/sample_submission.csv')
    sub['scalar_coupling_constant'] = prediction
    sub.to_csv('submission.csv', index=False)

def ensemble(args):
    base_df = pd.read_csv(args.csv_paths[0])
    for csv_path in args.csv_paths:
        sub_df = pd.read_csv(csv_path)
        base_df['scalar_coupling_constant'] += \
            sub_df['scalar_coupling_constant']
    
    base_df['scalar_coupling_constant'] /= len(args.csv_paths)
    base_df.to_csv('{}.csv'.format(args.ensemble_name), index=False)

if __name__ == '__main__':
    main()