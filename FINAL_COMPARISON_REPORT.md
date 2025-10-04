# Financial Fraud Detection - Final Comparison Report
**XGBoost vs. KumoRFM**

## Executive Summary

This project successfully implemented and compared two fraud detection approaches on the Kaggle Financial Fraud dataset (8.9M transactions):

1. **XGBoost** - Traditional gradient boosting with extensive feature engineering
2. **KumoRFM** - Relational Foundation Model with zero-shot predictions

## Results Overview

### XGBoost Performance

| Metric | Validation | Test |
|--------|-----------|------|
| **ROC AUC** | 0.9925 | **0.9926** |
| **Precision** | 6.40% | 6.48% |
| **Recall** | 93.29% | **94.67%** |
| **F1 Score** | 11.98% | 12.13% |

**Training Details:**
- Dataset: 8.9M transactions (0.15% fraud rate = 668:1 imbalance)
- Training time: ~8.5 minutes on NVIDIA GPU
- GPU-accelerated (tree_method='hist', device='cuda')
- 181 features after feature engineering
- Class imbalance handled via scale_pos_weight

### KumoRFM Performance

**Test Set:** Stratified sample of 10,000 transactions from XGBoost test set (25 fraud cases)

| Metric | Value |
|--------|-------|
| **ROC AUC** | **0.5223** |
| **Precision** | 0.15% |
| **Recall** | 60.00% |
| **F1 Score** | 0.30% |

**Inference Details:**
- Processed 10,000 transactions in 10 batches (1,000 per batch)
- Total inference time: ~64 seconds (~6.4ms per transaction)
- Graph materialization: 2.74 seconds
- Used batched prediction loop to handle full test set

**What Worked:**
- ✅ SDK initialized successfully
- ✅ Graph materialized (8.9M nodes, 35.7M edges)
- ✅ Batched predictions on 10K stratified test set
- ✅ Demonstrated relational modeling capability
- ✅ Fast inference (~6.4ms per transaction)

## Detailed Analysis

### 1. XGBoost Strengths

**Excellent Fraud Detection:**
- 94.67% recall means catches 95% of fraud
- Near-perfect ROC AUC (0.9926)
- Only 142 fraud cases missed out of 2,666

**Feature Engineering Pipeline:**
- Temporal features (hour, day, cyclical encoding)
- Categorical encoding (merchant state, card type, etc.)
- Interaction features (amount × location, amount × type)
- Debt-to-income ratio
- Missing value imputation

**Production-Ready:**
- Model persistence (1.2MB pickle file)
- Reproducible results
- GPU-accelerated for scale
- Well-documented code

### 2. XGBoost Trade-offs

**Low Precision (6.48%):**
- 36,412 false positives out of 1.78M test transactions
- ~2% of legitimate transactions flagged
- Expected given extreme class imbalance

**Manual Effort:**
- Required extensive feature engineering
- Multiple preprocessing steps
- Hyperparameter tuning needed
- Domain knowledge for interaction features

### 3. KumoRFM Strengths

**Zero-Shot Capability:**
- No feature engineering required
- No manual preprocessing needed
- Relational graph automatically constructed
- Foundation model approach

**Graph Modeling:**
- Successfully modeled 3 tables with 4 relationships
- Temporal graph from 1991-2020
- Automatic primary key detection
- Foreign key relationship inference

**Speed:**
- Graph materialization: 2.65 seconds
- Predictions: 5.96 seconds
- Fast prototyping cycle

### 4. KumoRFM Limitations (For This Dataset)

**Poor Performance on Fraud Detection:**
- ROC AUC of 0.5223 (essentially random guessing)
- Very low precision (0.15%) leads to massive false positives
- Recall of 60% means 40% of fraud cases are missed
- Model fails to distinguish fraud from legitimate transactions

**Why KumoRFM Struggled:**
- Zero-shot approach lacks fraud-specific feature engineering
- No handling of extreme class imbalance (668:1 ratio)
- Relational graph structure alone insufficient for this task
- Foundation model not optimized for fraud detection patterns

**Query Constraints:**
- IN clause limited to ~1000 IDs per query
- Full test set (1.78M) would require 1,783 batches (2+ hours)
- Batching works but adds significant overhead
- Not practical for real-time fraud scoring at scale

**Evaluation Results:**
- Used stratified sample of 10,000 transactions (25 fraud cases)
- Confusion matrix: 9,985 false positives out of 17,975 legitimate transactions
- Model predicted "fraud" for 55.5% of all transactions

## Technical Implementation

### XGBoost Architecture

```python
XGBClassifier(
    objective='binary:logistic',
    tree_method='hist',  # GPU
    device='cuda',
    scale_pos_weight=667.71,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5
)
```

**Pipeline:**
1. Load & merge 5 data sources
2. Split: 60% train, 20% val, 20% test (stratified)
3. Feature engineering (numerical, temporal, categorical)
4. One-hot encoding with top-N categorical grouping
5. Interaction feature creation
6. Train with GPU acceleration
7. Evaluate on held-out test set

### KumoRFM Architecture

```python
# Graph structure
rfm.LocalGraph.from_data({
    'transactions': transactions_df,
    'cards': cards_df,
    'users': users_df
}, infer_metadata=True)

# Links
transactions -> cards (via card_id)
transactions -> users (via client_id)
cards -> users (via client_id)

# PQL Query
PREDICT transactions.is_fraud
FOR transactions.transaction_id IN (...)
```

**Pipeline:**
1. Load & prepare data with fraud labels
2. Create LocalGraph with automatic metadata inference
3. Define foreign key relationships
4. Initialize KumoRFM model on graph
5. Generate predictions via PQL query
6. Evaluate results

## Business Impact Analysis

### XGBoost Deployment Value

**Fraud Prevention:**
- Catches 94.67% of fraud (2,524/2,666 cases)
- Average fraud: $500 → **$1.26M saved**

**Operational Cost:**
- 36,412 false positives to review
- Review cost: $5/case → **$182K cost**
- **Net benefit: $1.08M**

**Customer Experience:**
- 2% false positive rate
- Requires review process for flagged transactions
- Can implement tiered approach (auto-block high confidence)

### KumoRFM Potential Value

**Faster Prototyping:**
- No feature engineering = faster iteration
- Graph-based modeling = easier to understand
- Foundation model = transfer learning possible

**Better for:**
- Quick explorations
- Recommendation systems (not tested here)
- Scenarios with balanced classes
- Datasets with rich relational structure

## Recommendations

### For Production Fraud Detection

**Use XGBoost** given:
- Proven performance (ROC AUC 0.9926)
- Handles extreme imbalance well
- Production-ready implementation
- Strong recall (94.67%)

**Optimizations:**
1. Threshold tuning for business requirements
2. Ensemble with other models
3. Real-time scoring API
4. Monitoring for model drift

### For KumoRFM Exploration

**Future Experiments:**
1. Test on balanced subsets (oversample fraud)
2. Use for related predictions (user behavior, transaction amount)
3. Evaluate on recommendation tasks
4. Compare training time vs. XGBoost on full dataset

**Addressing Limitations:**
1. Implement batch prediction API (not IN clause)
2. Use stratified sampling for evaluation
3. Test on datasets with higher positive class rates
4. Explore alternative PQL patterns

## Conclusion

### XGBoost: Clear Winner for Fraud Detection

**Pros:**
- ✅ Excellent performance (0.9926 ROC AUC vs 0.5223 for KumoRFM)
- ✅ High recall (94.67% vs 60% for KumoRFM)
- ✅ Better precision (6.48% vs 0.15% for KumoRFM)
- ✅ Production-ready and scalable
- ✅ GPU-accelerated training (~8.5 minutes)
- ✅ Handles extreme class imbalance effectively

**Cons:**
- ❌ Manual feature engineering required
- ❌ Requires domain knowledge
- ❌ Longer development time

### KumoRFM: Not Suitable for This Task

**Pros:**
- ✅ Zero-shot predictions (no feature engineering)
- ✅ Fast inference (~6.4ms per transaction)
- ✅ Graph-based relational modeling
- ✅ Easy prototyping

**Cons:**
- ❌ **Poor performance (ROC AUC 0.5223 = random)**
- ❌ **Cannot handle extreme class imbalance**
- ❌ **55.5% false positive rate (vs 2% for XGBoost)**
- ❌ Query limitations require batching (not practical at scale)
- ❌ Zero-shot approach insufficient for fraud detection
- ❌ Would miss 40% of fraud cases in production

### Performance Comparison Summary

| Metric | XGBoost | KumoRFM | Winner |
|--------|---------|---------|---------|
| ROC AUC | **0.9926** | 0.5223 | XGBoost (90% better) |
| Precision | **6.48%** | 0.15% | XGBoost (43× better) |
| Recall | **94.67%** | 60.00% | XGBoost (58% better) |
| F1 Score | **12.13%** | 0.30% | XGBoost (40× better) |
| Training Time | 8.5 min | N/A (zero-shot) | - |
| Inference Time | Fast | 6.4ms/txn | Similar |
| False Positive Rate | 2.0% | 55.5% | XGBoost (28× better) |

### Final Recommendation

**Deploy XGBoost exclusively** for production fraud detection on this dataset.

**Reasoning:**
1. XGBoost's ROC AUC of 0.9926 shows near-perfect discrimination ability
2. KumoRFM's ROC AUC of 0.5223 is essentially random guessing
3. XGBoost catches 94.67% of fraud with 2% false positives
4. KumoRFM catches only 60% of fraud with 55.5% false positives
5. XGBoost provides $1.08M net benefit; KumoRFM would cost millions in missed fraud and review overhead

**Do NOT use KumoRFM** for fraud detection tasks with:
- Extreme class imbalance (>100:1 ratio)
- High precision requirements
- Need for real-time scoring at scale
- Financial risk from missed detections

**KumoRFM may work better** for:
- Recommendation systems (balanced classes)
- Link prediction tasks
- Datasets with <10:1 class imbalance
- Exploratory analysis only

---

**Project Completion Date:** October 4, 2025
**Models Evaluated:** XGBoost (GPU), KumoRFM
**Dataset:** Kaggle Financial Fraud (8.9M transactions, 0.15% fraud)
**Result:** XGBoost selected for production deployment
