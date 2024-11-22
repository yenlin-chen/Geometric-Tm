# coding: utf-8

import os

self_dir = os.path.dirname(os.path.normpath(__file__))

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(self_dir)))

data_dir = os.path.join(root_dir, 'data')

external_dir = os.path.join(data_dir, 'external')
collation_dir = os.path.join(data_dir, 'collation')
processed_dir = os.path.join(data_dir, 'processed')

deepstabp_col = os.path.join(collation_dir, 'DeepSTABp')

# one-hot-encoding for residue type
res_to_1hot = {
    'D': 0,
    'E': 1,
    'K': 2,
    'R': 3,
    'H': 4,
    'S': 5,
    'T': 6,
    'Y': 7,
    'N': 8,
    'Q': 9,
    'G': 10,
    'A': 11,
    'V': 12,
    'L': 13,
    'I': 14,
    'M': 15,
    'C': 16,
    'F': 17,
    'W': 18,
    'P': 19,
}

ohe_to_res = {v: k for k, v in res_to_1hot.items()}
