import argparse
import pandas as pd

from pathlib import Path

from .dataset import load_dataset, feature_engineering, split_fold

def arg_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'predict', 'split_fold'])
    arg('run_root')
    arg('--model', choices=['lgb', 'xgb'])
    arg('--n_fold', type=int, default= 5)
    arg('--fold', type=int, default=0)

    args = parser.parse_args()
    return args

def split(n_fold):
    train_df, test_df = load_dataset()
    train_df = split_fold(train_df, n_fold)
    train_df.to_csv('./train.csv')
    test_df.to_csv('./test.csv')

def main():
    args = arg_parser()
    if args.mode == 'split_fold':
        return split(args.n_fold)


    folds = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    train_fold = folds[folds['fold'] != args.fold]
    valid_fold = folds[folds['fold'] == args.fold]

    run_root = Path(args.run_root)

def train();
    pass

if __name__ == '__main__':
    main()