import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from .configs import csv_path


def load_dataset():
    train_df = pd.read_csv(csv_path['train'])
    test_df = pd.read_csv(csv_path['test'])

    structure_df = pd.read_csv(csv_path['structure'])

    train_df, test_df = feature_engineering(train_df, test_df, structure_df)

    # scalar_coupling_df = pd.read_csv(csv_path['scalar_coupling'])
    # potential_energy_df = pd.read_csv(csv_path['potential_energy'])
    # mulliken_charges_df = pd.read_csv(csv_path['mulliken_charges'])
    # magnetic_shielding_df = pd.read_csv(csv_path['magnetic_shielding'])
    # dipole_moments_df = pd.read_csv(csv_path['dipole_moments'])

    return train_df, test_df

def feature_engineering(train_df, test_df, structures):
    def map_atom_info(df, atom_idx):
        df = pd.merge(df, structures, how = 'left',
                    left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                    right_on = ['molecule_name',  'atom_index'])
        
        df = df.drop('atom_index', axis=1)
        df = df.rename(columns={'atom': f'atom_{atom_idx}',
                                'x': f'x_{atom_idx}',
                                'y': f'y_{atom_idx}',
                                'z': f'z_{atom_idx}'})
        return df

    train_df = map_atom_info(train_df, 0)
    train_df = map_atom_info(train_df, 1)

    test_df = map_atom_info(test_df, 0)
    test_df = map_atom_info(test_df, 1)
    train_df_p_0 = train_df[['x_0', 'y_0', 'z_0']].values
    train_df_p_1 = train_df[['x_1', 'y_1', 'z_1']].values
    test_df_p_0 = test_df[['x_0', 'y_0', 'z_0']].values
    test_df_p_1 = test_df[['x_1', 'y_1', 'z_1']].values
    train_df['dist'] = np.linalg.norm(train_df_p_0 - train_df_p_1, axis=1)
    test_df['dist'] = np.linalg.norm(test_df_p_0 - test_df_p_1, axis=1)
    train_df['dist_x'] = (train_df['x_0'] - train_df['x_1']) ** 2
    test_df['dist_x'] = (test_df['x_0'] - test_df['x_1']) ** 2
    train_df['dist_y'] = (train_df['y_0'] - train_df['y_1']) ** 2
    test_df['dist_y'] = (test_df['y_0'] - test_df['y_1']) ** 2
    train_df['dist_z'] = (train_df['z_0'] - train_df['z_1']) ** 2
    test_df['dist_z'] = (test_df['z_0'] - test_df['z_1']) ** 2

    train_df['type_0'] = train_df['type'].apply(lambda x: x[0])
    test_df['type_0'] = test_df['type'].apply(lambda x: x[0])
    train_df['type_1'] = train_df['type'].apply(lambda x: x[1:])
    test_df['type_1'] = test_df['type'].apply(lambda x: x[1:])

    for f in ['atom_0', 'atom_1', 'type_0', 'type_1', 'type']:
        lbl = LabelEncoder()
        lbl.fit(list(train_df[f].values) + list(test_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values)).astype('int')
        test_df[f] = lbl.transform(list(test_df[f].values)).astype('int')
    
    return train_df, test_df

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