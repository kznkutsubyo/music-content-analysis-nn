# Training and Evaluation

This directory contains training and evaluation scripts used for the GTZAN experiments.

The original dataset and trained checkpoints are not included in this public repository.

Typical workflow:

1. Prepare GTZAN train/validation/test splits.
2. Train classical models based on MFCC features.
3. Train CNN and AST models on log-mel representations.
4. Evaluate all models and export metrics/confusion matrices.

See the scripts in `src/` for the implementation details.
