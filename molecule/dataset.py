import pandas as pd

from sklearn.model_selection import KFold

from .configs import csv_path

def load_dataset():
    train_df = pd.read_csv(csv_path['train'])
    test_df = pd.read_csv(csv_path['test'])

    # structure_df = pd.read_csv(csv_path['structure'])
    # scalar_coupling_df = pd.read_csv(csv_path['scalar_coupling'])
    # potential_energy_df = pd.read_csv(csv_path['potential_energy'])
    # mulliken_charges_df = pd.read_csv(csv_path['mulliken_charges'])
    # magnetic_shielding_df = pd.read_csv(csv_path['magnetic_shielding'])
    # dipole_moments_df = pd.read_csv(csv_path['dipole_moments'])
    return train_df, test_df

def feature_engineering(df):
    return df

def split_fold(df, n_fold=5):
    """
        dataframe을 n_fold에 따라서 'fold' column에 fold를 할당합니다.
        - arguments
            - df: fold를 나누고자하는 dataframe
            - n_fold: fold의 수
        - return
            - df: 'fold' column이 추가된 dataframe
    """
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=11)
    df_folds = list()

    for i, result in enumerate(kf.split(df)):
        sub_df = df.loc[result[1]]
        sub_df['fold'] = i
        df_folds.append(sub_df)

    df = pd.concat(df_folds).sort_index()
    
    return df