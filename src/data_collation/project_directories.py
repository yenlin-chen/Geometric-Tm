import os

self_dir = os.path.dirname(__file__)

root_dir = os.path.normpath(self_dir + '/..' *2)
data_dir = os.path.join(root_dir, 'data')

external_dir = os.path.join(data_dir, 'external')
collation_dir = os.path.join(data_dir, 'collation')

deepstabp_ext = os.path.join(
    external_dir, 'DeepSTABp', 'Melting_temperatures_of_proteins'
)

deepstabp_col = os.path.join(collation_dir, 'DeepSTABp')
