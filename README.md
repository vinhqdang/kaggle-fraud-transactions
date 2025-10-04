# Financial Fraud Detection

This project implements fraud detection using two approaches:
1. **XGBoost** - Traditional gradient boosting with extensive feature engineering
2. **Kumo AI** - Relational foundation model leveraging graph-based data

## Dataset

Source: [Kaggle Transactions Fraud Dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data)

The dataset contains:
- `transactions_data.csv` - Transaction records with amounts, timestamps, merchant details
- `cards_data.csv` - Credit/debit card information
- `users_data.csv` - Customer demographic and financial data
- `mcc_codes.json` - Merchant category codes
- `train_fraud_labels.json` - Fraud labels for transactions

## Project Structure

```
.
├── config.py                      # Configuration file with API keys
├── data/                          # Data directory
│   ├── transactions_data.csv
│   ├── cards_data.csv
│   ├── users_data.csv
│   ├── mcc_codes.json
│   └── train_fraud_labels.json
├── xgboost_fraud_detection.py    # XGBoost implementation with GPU support
├── kumo_fraud_detection.py       # Kumo AI implementation
├── compare_models.py              # Model comparison script
├── models/                        # Saved models
├── results/                       # Results and comparison reports
└── requirements.txt               # Python dependencies
```

## Setup Instructions

### 1. Environment Setup

Create and activate conda environment:

```bash
# Create conda environment with Python 3.10
conda create -n py310 python=3.10 -y

# Activate environment
conda activate py310
```

### 2. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

**For GPU support with XGBoost**, ensure you have CUDA installed and XGBoost is configured to use GPU.

### 3. Configuration

The API key for Kumo AI is already configured in `config.py`. Update if needed:

```python
KUMO_API_KEY = "your_api_key_here"
KUMO_API_URL = "https://api.kumo.ai/api"
```

## Running the Code

### Option 1: Run All Models and Compare (Recommended)

Run everything with a single command:

```bash
# Run XGBoost, then Kumo, then comparison
python xgboost_fraud_detection.py && python kumo_fraud_detection.py && python compare_models.py
```

### Option 2: Run Individual Models

**Run XGBoost model:**

```bash
python xgboost_fraud_detection.py
```

This will:
- Load and merge all data files
- Apply comprehensive feature engineering
- Train XGBoost model with GPU support (tree_method='hist', device='cuda')
- Evaluate on validation and test sets
- Save model to `models/xgboost_fraud_model.pkl`
- Save results to `results/xgboost_results.csv`

**Run Kumo AI model:**

```bash
python kumo_fraud_detection.py
```

This will:
- Initialize Kumo SDK with API key
- Load and prepare relational data
- Build graph connecting transactions, cards, users, and MCC codes
- Train Kumo's relational foundation model
- Make predictions and evaluate
- Save results to `results/kumo_results.csv`

**Compare models:**

```bash
python compare_models.py
```

This will:
- Load results from both models
- Generate comparison table
- Create visualization plots
- Generate detailed comparison report
- Save outputs to `results/` directory

## Results

After running all models, you'll find:

- `results/xgboost_results.csv` - XGBoost performance metrics
- `results/kumo_results.csv` - Kumo AI performance metrics
- `results/model_comparison.csv` - Side-by-side comparison
- `results/model_comparison.png` - Visualization of metrics
- `results/comparison_report.txt` - Detailed analysis and recommendations

## Algorithm Overview

### XGBoost Approach

**Feature Engineering:**
- Numerical feature cleaning (amounts, income, credit limits)
- Date feature extraction (hour, day of week, month, days to expiry)
- Cyclical encoding (sin/cos transformations for time features)
- Binary feature mapping (gender, has_chip, has_error)
- One-hot encoding for categorical features (merchant_state, card_type, etc.)
- Interaction features (amount × merchant_state, amount × transaction_type, etc.)
- Debt-to-income ratio calculation

**Model Configuration:**
- GPU-accelerated training (tree_method='hist', device='cuda')
- Scale_pos_weight for class imbalance handling
- 60/20/20 train/validation/test split with stratification
- Hyperparameters: n_estimators=500, learning_rate=0.05, max_depth=5

**Key Features:**
- Merchant location (zip, state)
- Transaction type (online, chip, swipe)
- Merchant descriptions (tolls, taxis, etc.)
- Amount-based interactions
- Temporal patterns

### Kumo AI Approach

**Graph Structure:**
- Nodes: Transactions, Cards, Users, MCC Codes
- Edges: Transaction→Card, Transaction→User, Transaction→MCC
- Leverages relational structure without manual feature engineering

**Model:**
- Relational Foundation Model (RFM)
- Predictive Query Language (PQL) for defining prediction task
- Automatic feature learning from graph structure
- Direct predictions from relational data

**Advantages:**
- No manual feature engineering required
- Faster prototyping and iteration
- Handles relational data naturally
- Automatic feature extraction from graph

## Performance Metrics

Both models are evaluated on:
- **ROC AUC** - Overall discrimination ability
- **Precision** - Accuracy of fraud predictions
- **Recall** - Coverage of actual fraud cases
- **F1 Score** - Harmonic mean of precision and recall

## GPU Requirements

**XGBoost GPU Support:**
- CUDA-capable GPU (NVIDIA)
- CUDA toolkit installed
- XGBoost configured with GPU support

To verify GPU is being used:
```python
import xgboost as xgb
print(xgb.device)  # Should show 'cuda'
```

## Troubleshooting

**Kumo SDK Installation Issues:**
```bash
pip install kumo --upgrade
```

**GPU Not Detected:**
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA-enabled XGBoost
pip install xgboost --upgrade
```

**Memory Issues:**
```bash
# Reduce sample size in xgboost_fraud_detection.py
SAMPLE_SIZE_FOR_TUNING = 1000000  # Reduce this value
```

## Development Notes

**Code Design:**
- Object-oriented approach with clear class structures
- Modular methods for each pipeline step
- Comprehensive error handling and logging
- Memory-efficient data processing (downcasting, garbage collection)

**Best Practices:**
- Stratified sampling for class imbalance
- Separate preprocessing for train/val/test (no data leakage)
- Model persistence with pickle
- Reproducible results with random seeds

## Next Steps

1. **Hyperparameter Tuning:** Run RandomizedSearchCV on full dataset
2. **Ensemble Methods:** Combine XGBoost and Kumo predictions
3. **Feature Selection:** Remove low-importance features
4. **Threshold Optimization:** Adjust prediction threshold based on business costs
5. **Temporal Validation:** Test on time-based splits for production readiness
6. **Model Monitoring:** Implement drift detection for production deployment

## References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- Kumo AI Documentation: https://kumo-ai.github.io/kumo-sdk/
- Kaggle Dataset: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/

## License

This project is for educational and research purposes.
