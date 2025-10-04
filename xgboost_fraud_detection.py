"""
XGBoost Fraud Detection Implementation
Reimplemented from Kaggle notebook to run locally with GPU support
"""

import numpy as np
import pandas as pd
import json
import gc
import re
import os
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_DIR, RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE


class XGBoostFraudDetector:
    """XGBoost-based fraud detection model with comprehensive preprocessing"""

    def __init__(self, random_state=RANDOM_SEED):
        self.random_state = random_state
        self.model = None
        self.encoder = None
        self.median_imputations = None
        self.top_fraud_states = None
        self.feature_names = None
        self.results = {}

    def load_data(self):
        """Load all data files"""
        print("Loading data files...")

        # Load fraud labels
        with open(os.path.join(DATA_DIR, 'train_fraud_labels.json'), 'r') as f:
            raw_json_data = json.load(f)
        transaction_labels_dict = raw_json_data['target']
        train_fraud_labels = pd.Series(transaction_labels_dict).reset_index()
        train_fraud_labels.columns = ['transaction_id', 'is_fraud']
        train_fraud_labels['transaction_id'] = pd.to_numeric(train_fraud_labels['transaction_id'])

        # Load other files
        transaction_df = pd.read_csv(os.path.join(DATA_DIR, 'transactions_data.csv'))
        card_df = pd.read_csv(os.path.join(DATA_DIR, 'cards_data.csv'))
        users_df = pd.read_csv(os.path.join(DATA_DIR, 'users_data.csv'))
        mcc_series = pd.read_json(os.path.join(DATA_DIR, 'mcc_codes.json'), typ='series')
        mcc_df = mcc_series.reset_index()
        mcc_df.columns = ['mcc_code', 'description']

        print("All data files loaded successfully.")
        return transaction_df, train_fraud_labels, card_df, users_df, mcc_df

    def merge_data(self, transaction_df, train_fraud_labels, card_df, users_df, mcc_df):
        """Merge all dataframes"""
        print("Merging all dataframes...")

        df = pd.merge(transaction_df, train_fraud_labels, left_on='id', right_on='transaction_id', how='left')
        df = pd.merge(df, card_df, left_on='card_id', right_on='id', how='left', suffixes=('', '_card'))
        df = pd.merge(df, users_df, left_on='client_id', right_on='id', how='left', suffixes=('', '_user'))
        df = pd.merge(df, mcc_df, left_on='mcc', right_on='mcc_code', how='left')

        df = df.drop(columns=['transaction_id', 'id_card', 'id_user', 'mcc_code'])

        print(f"Merged dataframe shape: {df.shape}")
        return df

    def split_data(self, df):
        """Split data into train/validation/test sets"""
        print("Splitting data...")

        # Drop rows with missing fraud labels
        df.dropna(subset=['is_fraud'], inplace=True)
        df['is_fraud'] = df['is_fraud'].map({'No': 0, 'Yes': 1})

        features = [col for col in df.columns if col != 'is_fraud']
        X = df[features]
        y = df['is_fraud']

        # 60/20/20 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.40, random_state=self.random_state, stratify=y
        )
        X_cv, X_test, y_cv, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.random_state, stratify=y_temp
        )

        print(f"X_train shape: {X_train.shape}")
        print(f"X_cv shape: {X_cv.shape}")
        print(f"X_test shape: {X_test.shape}")

        return X_train, X_cv, X_test, y_train, y_cv, y_test

    def apply_preprocessing(self, df, is_training_set=False):
        """Apply feature engineering and preprocessing"""
        df_processed = df.copy()

        # Clean numerical columns
        amount_cols = ['amount', 'per_capita_income', 'yearly_income', 'credit_limit', 'total_debt']
        for col in amount_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(
                    df_processed[col].astype(str).str.replace(r'[$,]', '', regex=True),
                    errors='coerce'
                )

        # Date engineering
        date_cols = ['date', 'expires', 'acct_open_date']
        for col in date_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce', format='mixed')

        if 'date' in df_processed.columns:
            df_processed['hour_of_day'] = df_processed['date'].dt.hour
            df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
            df_processed['month'] = df_processed['date'].dt.month

        if 'expires' in df_processed.columns and 'date' in df_processed.columns:
            df_processed['days_to_expiry'] = (df_processed['expires'] - df_processed['date']).dt.days

        df_processed.drop(columns=date_cols, inplace=True, errors='ignore')

        # Cyclical features
        cyclical_cols = ['hour_of_day', 'day_of_week', 'month']
        if all(col in df_processed.columns for col in cyclical_cols):
            df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour_of_day'] / 24.0)
            df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour_of_day'] / 24.0)
            df_processed['day_of_week_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7.0)
            df_processed['day_of_week_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7.0)
            df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12.0)
            df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12.0)
            df_processed.drop(columns=cyclical_cols, inplace=True)

        # Binary features
        if 'errors' in df_processed.columns:
            df_processed['has_error'] = df_processed['errors'].notna().astype(int)
        if 'gender' in df_processed.columns:
            df_processed['gender'] = df_processed['gender'].map({'Female': 0, 'Male': 1})
        if 'has_chip' in df_processed.columns:
            df_processed['has_chip'] = df_processed['has_chip'].map({'NO': 0, 'YES': 1})

        # Numerical imputation
        numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
        if is_training_set:
            self.median_imputations = df_processed[numerical_cols].median()

        if self.median_imputations is not None:
            df_processed.fillna(self.median_imputations, inplace=True)

        return df_processed

    def drop_unnecessary_columns(self, X_train, X_cv, X_test):
        """Drop unnecessary columns"""
        cols_to_drop = [
            'id', 'client_id', 'card_id', 'merchant_id', 'card_number', 'cvv', 'mcc',
            'acct_open_date', 'year_pin_last_changed', 'card_on_dark_web', 'has_chip',
            'address', 'merchant_city', 'birth_year', 'birth_month', 'latitude', 'longitude',
            'date', 'expires'
        ]

        for df_set in [X_train, X_cv, X_test]:
            cols_that_exist = [col for col in cols_to_drop if col in df_set.columns]
            df_set.drop(columns=cols_that_exist, inplace=True, errors='ignore')

        return X_train, X_cv, X_test

    def encode_categorical(self, X_train, X_cv, X_test, y_train):
        """Group and one-hot encode categorical features"""
        print("Encoding categorical features...")

        # Group merchant_state
        if 'merchant_state' in X_train.columns:
            temp_train_df = pd.DataFrame({'merchant_state': X_train['merchant_state'], 'is_fraud': y_train})
            fraud_counts = temp_train_df[temp_train_df['is_fraud'] == 1]['merchant_state'].value_counts()
            self.top_fraud_states = fraud_counts.nlargest(15).index.tolist()

            for df_set in [X_train, X_cv, X_test]:
                df_set.loc[:, 'merchant_state'] = df_set['merchant_state'].apply(
                    lambda x: x if x in self.top_fraud_states else 'OTHER_STATE'
                )

        # One-hot encode
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        print(f'Categorical columns to encode: {categorical_cols}')

        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int8)
        self.encoder.fit(X_train[categorical_cols])

        encoded_train = pd.DataFrame(
            self.encoder.transform(X_train[categorical_cols]),
            index=X_train.index,
            columns=self.encoder.get_feature_names_out(categorical_cols)
        )
        encoded_cv = pd.DataFrame(
            self.encoder.transform(X_cv[categorical_cols]),
            index=X_cv.index,
            columns=self.encoder.get_feature_names_out(categorical_cols)
        )
        encoded_test = pd.DataFrame(
            self.encoder.transform(X_test[categorical_cols]),
            index=X_test.index,
            columns=self.encoder.get_feature_names_out(categorical_cols)
        )

        X_train = pd.concat([X_train.drop(columns=categorical_cols), encoded_train], axis=1)
        X_cv = pd.concat([X_cv.drop(columns=categorical_cols), encoded_cv], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), encoded_test], axis=1)

        return X_train, X_cv, X_test

    def add_interaction_features(self, X_train, X_cv, X_test):
        """Add interaction features"""
        print("Adding interaction features...")

        for df_set in [X_train, X_cv, X_test]:
            if all(col in df_set.columns for col in ['amount', 'merchant_state_Italy']):
                df_set['amount_x_state_italy'] = df_set['amount'] * df_set['merchant_state_Italy']
            if all(col in df_set.columns for col in ['amount', 'description_Tolls and Bridge Fees']):
                df_set['amount_x_tolls'] = df_set['amount'] * df_set['description_Tolls and Bridge Fees']
            if all(col in df_set.columns for col in ['amount', 'use_chip_Online Transaction']):
                df_set['amount_x_online_trans'] = df_set['amount'] * df_set['use_chip_Online Transaction']
            if all(col in df_set.columns for col in ['total_debt', 'yearly_income']):
                df_set['debt_to_income_ratio'] = df_set['total_debt'] / (df_set['yearly_income'] + 1e-6)

        return X_train, X_cv, X_test

    def downcast_dtypes(self, X_train, X_cv, X_test):
        """Downcast data types for memory efficiency"""
        for df_set in [X_train, X_cv, X_test]:
            for col in df_set.select_dtypes(include=['float64', 'int64']).columns:
                if 'float' in str(df_set[col].dtype):
                    df_set.loc[:, col] = df_set[col].astype('float32')
                else:
                    df_set.loc[:, col] = pd.to_numeric(df_set[col], downcast='integer')

        return X_train, X_cv, X_test

    def train_model(self, X_train, y_train, X_cv, y_cv, use_gpu=True):
        """Train XGBoost model with GPU support"""
        print("Training XGBoost model...")

        # Calculate scale_pos_weight
        neg_count = y_train.value_counts()[0]
        pos_count = y_train.value_counts()[1]
        scale_pos_weight = neg_count / pos_count
        print(f"Scale pos weight: {scale_pos_weight:.2f}")

        # Model parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',  # Use 'hist' for GPU
            'device': 'cuda' if use_gpu else 'cpu',
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.random_state,
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 0.1
        }

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist()

        # Evaluate
        y_pred_cv = self.model.predict(X_cv)
        y_proba_cv = self.model.predict_proba(X_cv)[:, 1]

        print("\n=== Validation Set Results ===")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_cv, y_pred_cv))
        print("\nClassification Report:")
        print(classification_report(y_cv, y_pred_cv))
        roc_auc = roc_auc_score(y_cv, y_proba_cv)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

        # Store results
        self.results['validation'] = {
            'roc_auc': roc_auc,
            'precision': precision_score(y_cv, y_pred_cv),
            'recall': recall_score(y_cv, y_pred_cv),
            'f1': f1_score(y_cv, y_pred_cv)
        }

        return self.model

    def evaluate_test(self, X_test, y_test):
        """Evaluate on test set"""
        print("\n=== Test Set Results ===")

        y_pred_test = self.model.predict(X_test)
        y_proba_test = self.model.predict_proba(X_test)[:, 1]

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_test))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))
        roc_auc = roc_auc_score(y_test, y_proba_test)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

        # Store results
        self.results['test'] = {
            'roc_auc': roc_auc,
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test)
        }

        return self.results

    def save_model(self, filepath='xgboost_model.pkl'):
        """Save the trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'encoder': self.encoder,
                'median_imputations': self.median_imputations,
                'top_fraud_states': self.top_fraud_states,
                'feature_names': self.feature_names,
                'results': self.results
            }, f)
        print(f"Model saved to {filepath}")

    def run_full_pipeline(self, use_gpu=True):
        """Run the complete fraud detection pipeline"""
        print("=" * 80)
        print("XGBOOST FRAUD DETECTION PIPELINE")
        print("=" * 80)

        # Load and merge data
        transaction_df, train_fraud_labels, card_df, users_df, mcc_df = self.load_data()
        df = self.merge_data(transaction_df, train_fraud_labels, card_df, users_df, mcc_df)

        # Clean up memory
        del transaction_df, train_fraud_labels, card_df, users_df, mcc_df
        gc.collect()

        # Split data
        X_train, X_cv, X_test, y_train, y_cv, y_test = self.split_data(df)
        del df
        gc.collect()

        # Preprocessing
        print("\n=== Preprocessing ===")
        X_train = self.apply_preprocessing(X_train, is_training_set=True)
        X_cv = self.apply_preprocessing(X_cv)
        X_test = self.apply_preprocessing(X_test)

        X_train, X_cv, X_test = self.drop_unnecessary_columns(X_train, X_cv, X_test)
        X_train, X_cv, X_test = self.encode_categorical(X_train, X_cv, X_test, y_train)
        X_train, X_cv, X_test = self.add_interaction_features(X_train, X_cv, X_test)
        X_train, X_cv, X_test = self.downcast_dtypes(X_train, X_cv, X_test)

        print(f"\nFinal X_train shape: {X_train.shape}")
        print(f"Final X_cv shape: {X_cv.shape}")
        print(f"Final X_test shape: {X_test.shape}")

        # Check for NaNs
        print("\n=== NaN Check ===")
        nan_train = X_train.isna().sum().sum()
        nan_cv = X_cv.isna().sum().sum()
        nan_test = X_test.isna().sum().sum()
        print(f"NaNs in X_train: {nan_train}")
        print(f"NaNs in X_cv: {nan_cv}")
        print(f"NaNs in X_test: {nan_test}")

        # Train model
        print("\n=== Training ===")
        self.train_model(X_train, y_train, X_cv, y_cv, use_gpu=use_gpu)

        # Evaluate on test set
        self.evaluate_test(X_test, y_test)

        # Save model
        self.save_model('models/xgboost_fraud_model.pkl')

        return self.results


def main():
    """Main execution function"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Initialize and run pipeline
    detector = XGBoostFraudDetector()
    results = detector.run_full_pipeline(use_gpu=True)

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results/xgboost_results.csv')
    print("\n=== Results saved to results/xgboost_results.csv ===")
    print(results_df)


if __name__ == "__main__":
    main()
