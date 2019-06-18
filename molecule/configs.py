import os

data_path = '../input'

csv_path = {
    'train': os.path.join(data_path, 'train.csv'), # id     molecule_name  atom_index_0  atom_index_1  type  scalar_coupling_constant
    'test': os.path.join(data_path, 'test.csv'), # id     molecule_name  atom_index_0  atom_index_1  type
    'structure': os.path.join(data_path, 'structures.csv'), # molecule_name  atom_index atom         x         y         z
    'scalar_coupling': os.path.join(data_path, 'scalar_coupling_contributions.csv'), # molecule_name  atom_index_0  atom_index_1  type       fc        sd      pso       dso
    'potential_energy': os.path.join(data_path, 'potential_energy.csv'), # molecule_name  potential_energy
    'mulliken_charges': os.path.join(data_path, 'mulliken_charges.csv'), # molecule_name  atom_index  mulliken_charge
    'magnetic_shielding': os.path.join(data_path, 'magnetic_shielding_tensors.csv'), # molecule_name  atom_index        XX      YX      ZX      XY        YY      ZY      XZ      YZ        ZZ
    'dipole_moments': os.path.join(data_path, 'dipole_moments.csv'), # molecule_name       X    Y       Z
}
