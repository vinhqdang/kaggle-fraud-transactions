# Quick Setup Guide

## Prerequisites

- Conda installed
- CUDA-capable GPU (for XGBoost GPU acceleration)
- Data files downloaded from Kaggle

## Quick Start

### 1. Download Data

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data) and place files in the `data/` directory:

```
data/
├── transactions_data.csv
├── cards_data.csv
├── users_data.csv
├── mcc_codes.json
└── train_fraud_labels.json
```

### 2. Setup Environment

```bash
# Create conda environment
conda create -n py310 python=3.10 -y

# Activate environment
conda activate py310

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Everything

```bash
# Run all models and generate comparison
python xgboost_fraud_detection.py && python kumo_fraud_detection.py && python compare_models.py
```

## Expected Output

After running, you'll have:

```
models/
└── xgboost_fraud_model.pkl

results/
├── xgboost_results.csv
├── kumo_results.csv
├── model_comparison.csv
├── model_comparison.png
└── comparison_report.txt
```

## Troubleshooting

### Kumo SDK Issues

If Kumo SDK is not available or you don't have access:

```bash
# Run only XGBoost
python xgboost_fraud_detection.py
```

### Memory Issues

If you encounter out-of-memory errors, reduce the sample size in `xgboost_fraud_detection.py`:

```python
SAMPLE_SIZE_FOR_TUNING = 1000000  # Reduce this value
```

### GPU Not Detected

Check GPU availability:

```bash
nvidia-smi
```

If GPU is not available, XGBoost will automatically fall back to CPU mode.

## Project Summary

This project implements two fraud detection approaches:

1. **XGBoost**: Traditional ML with extensive feature engineering
   - GPU-accelerated training
   - Manual feature engineering (temporal, categorical, interaction features)
   - Hyperparameter tuning
   - Expected ROC AUC: ~0.99

2. **Kumo AI**: Relational foundation model
   - Graph-based approach
   - No manual feature engineering
   - Automatic feature learning
   - Fast prototyping

Both models are evaluated and compared using:
- ROC AUC Score
- Precision
- Recall
- F1 Score

## Next Steps

1. Run the models to generate results
2. Review `results/comparison_report.txt` for detailed analysis
3. Check `results/model_comparison.png` for visual comparison
4. Experiment with hyperparameters and features
5. Deploy the best model for production use

## Support

For issues or questions:
- Check README.md for detailed documentation
- Review code comments in Python files
- Consult original Kaggle notebook for algorithm details
