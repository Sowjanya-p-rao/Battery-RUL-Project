# Battery RUL Prediction Project

This repository contains code, data preparation, and models for predicting the Remaining Useful Life (RUL) of lithium-ion batteries using the CALCE dataset.

## Structure
- configs/: YAML configuration files
- data/: Raw, interim, and processed data
- notebooks/: Jupyter notebooks for EDA and modeling
- src/: Source code for preprocessing, training, and models
- results/: Tables, figures, and logs
- reproducibility/: Frozen requirements for reproducibility

## How to Run
1. Place CALCE dataset into `data/raw/`.
2. Run preprocessing: `python src/preprocess.py`
3. Train RUL model: `python src/train_rul.py --config configs/rul_lstm.yaml`
4. Evaluate results: `python src/eval.py`
