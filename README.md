# Training and evaluation
Go to `experiments/sandbox_50 (example)/v7a-the_fold/contact_12` and run `train-10fold.py`.

This will initiated a pyg dataset object, which will proceed to download protein structures from AlphaFoldDB, run modal analysis with TNM, and build the graphs required.
Training will proceed after data processing.

# Evaluation
Examples are provided in `/experiments/sandbox_50 (example)/v7a-the_fold/contact_12/test_distr-DeepSTABp.py'

# File Hierarchy
## data
Protein structures downloaded from AlphaFoldDB are saved to `data/external/AlphaFoldDB`, and TNM results are saved to `data/collation/TNM`.
Processed graphs for the GNN are saved at `/data/processed/<custom_name>`, where then name can be specified in the training and inference script.
## experiments
This directory is meant to contain all runs and their results.
## src
Modules imported by the experiments are under `/src/ml_modules`.
Code for collating the Tm and OGT values are under `/src/data_collation`.

# Requirements
Python packages requried:
- pytorch
- pytorch-cuda=12.4
- pyg
- matplotlib
- gudhi
- tqdm
- requests
- prody
- sortednp
- torchinfo
- scikit-learn
- transformers
- sentencepiece
Additional dependencies:
- TNM (https://github.com/ugobas/tnm)
