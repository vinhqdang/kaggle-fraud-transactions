"""
KumoRFM Fraud Detection Implementation
Using KumoRFM (Kumo Relational Foundation Model) for fraud detection
Based on official KumoRFM quickstart notebook
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

try:
    import kumoai.experimental.rfm as rfm
except ImportError:
    print("Warning: KumoAI SDK not installed. Install with: pip install kumoai")
    rfm = None

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)

from config import DATA_DIR, KUMO_API_KEY, RANDOM_SEED


class KumoRFMFraudDetector:
    """KumoRFM-based fraud detection model"""

    def __init__(self, api_key=KUMO_API_KEY):
        self.api_key = api_key
        self.graph = None
        self.model = None
        self.results = {}

    def initialize_kumo(self):
        """Initialize KumoRFM SDK"""
        if rfm is None:
            raise ImportError("KumoAI SDK is not installed. Install with: pip install kumoai")

        print("Initializing KumoRFM SDK...")
        rfm.init(api_key=self.api_key)
        print("KumoRFM SDK initialized successfully")

    def load_and_prepare_data(self):
        """Load and prepare data for KumoRFM"""
        print("Loading data files...")

        # Load fraud labels
        with open(os.path.join(DATA_DIR, 'train_fraud_labels.json'), 'r') as f:
            raw_json_data = json.load(f)
        transaction_labels_dict = raw_json_data['target']
        fraud_labels_df = pd.DataFrame([
            {'transaction_id': int(k), 'is_fraud': 1 if v == 'Yes' else 0}
            for k, v in transaction_labels_dict.items()
        ])

        # Load transaction data
        transactions_df = pd.read_csv(os.path.join(DATA_DIR, 'transactions_data.csv'))

        # Merge fraud labels with transactions
        transactions_df = transactions_df.merge(
            fraud_labels_df,
            left_on='id',
            right_on='transaction_id',
            how='left'
        )

        # Only keep labeled transactions for training/evaluation
        transactions_df = transactions_df[transactions_df['is_fraud'].notna()].copy()

        # Ensure correct data types
        transactions_df['id'] = transactions_df['id'].astype(int)
        transactions_df['card_id'] = transactions_df['card_id'].astype(int)
        transactions_df['client_id'] = transactions_df['client_id'].astype(int)
        transactions_df['is_fraud'] = transactions_df['is_fraud'].astype(int)

        # Convert date to datetime
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])

        # Load cards data
        cards_df = pd.read_csv(os.path.join(DATA_DIR, 'cards_data.csv'))
        cards_df['id'] = cards_df['id'].astype(int)
        cards_df['client_id'] = cards_df['client_id'].astype(int)

        # Load users data
        users_df = pd.read_csv(os.path.join(DATA_DIR, 'users_data.csv'))
        users_df['id'] = users_df['id'].astype(int)

        print(f"Transactions shape: {transactions_df.shape}")
        print(f"Cards shape: {cards_df.shape}")
        print(f"Users shape: {users_df.shape}")
        print(f"Fraud rate: {transactions_df['is_fraud'].mean():.4f}")

        return transactions_df, cards_df, users_df

    def create_graph(self, transactions_df, cards_df, users_df):
        """Create KumoRFM graph from data"""
        print("Creating KumoRFM graph...")

        # Create LocalGraph from dataframes
        # This automatically creates LocalTable objects and infers metadata
        self.graph = rfm.LocalGraph.from_data({
            'transactions': transactions_df,
            'cards': cards_df,
            'users': users_df
        }, infer_metadata=True)

        # Print existing links (from_data already infers some)
        print("\nExisting graph structure:")
        self.graph.print_links()

        # Add any missing links if needed
        print("\nAdding additional links if needed...")

        # Get existing edges to avoid duplicates
        existing_edges = [(e.src_table, e.fkey, e.dst_table) for e in self.graph.edges]

        # Define desired links
        desired_links = [
            ("transactions", "card_id", "cards"),
            ("transactions", "client_id", "users"),
            ("cards", "client_id", "users")
        ]

        # Add links that don't exist yet
        for src, fkey, dst in desired_links:
            if (src, fkey, dst) not in existing_edges:
                try:
                    self.graph.link(src_table=src, fkey=fkey, dst_table=dst)
                    print(f"Added link: {src}.{fkey} -> {dst}")
                except Exception as e:
                    print(f"Skipping link {src}.{fkey} -> {dst}: {str(e)}")

        # Print final graph structure
        print("\nFinal graph structure:")
        self.graph.print_links()

        print("KumoRFM graph created successfully")
        return self.graph

    def train_and_predict(self, transactions_df, test_transaction_ids=None, max_samples=10000):
        """Train and make predictions using KumoRFM with batched predictions"""
        print("\nInitializing KumoRFM model...")

        # Create KumoRFM model
        self.model = rfm.KumoRFM(self.graph)

        print("Making predictions with KumoRFM...")

        # Load test set transaction IDs if not provided
        if test_transaction_ids is None:
            test_ids_path = 'results/test_transaction_ids.npy'
            if os.path.exists(test_ids_path):
                test_transaction_ids = np.load(test_ids_path)
                print(f"Loaded {len(test_transaction_ids)} test transaction IDs from {test_ids_path}")
            else:
                print("No test set IDs found, will sample randomly")
                sample_size = min(1000, len(transactions_df))
                sampled_transactions = transactions_df.sample(n=sample_size, random_state=42)
                test_transaction_ids = sampled_transactions['id'].values

        # Filter transactions to only those in test set
        test_transactions = transactions_df[transactions_df['id'].isin(test_transaction_ids)].copy()
        print(f"Full test set contains {len(test_transactions)} transactions")
        print(f"Fraud rate in full test set: {test_transactions['is_fraud'].mean():.4f}")

        # Use stratified sampling to get a manageable subset
        if len(test_transactions) > max_samples:
            print(f"\nUsing stratified sample of {max_samples} transactions for evaluation...")
            from sklearn.model_selection import train_test_split
            # Stratified sampling to maintain fraud rate
            test_transactions, _ = train_test_split(
                test_transactions,
                test_size=(len(test_transactions) - max_samples) / len(test_transactions),
                random_state=42,
                stratify=test_transactions['is_fraud']
            )
            print(f"Sampled test set contains {len(test_transactions)} transactions")
            print(f"Fraud rate in sampled test set: {test_transactions['is_fraud'].mean():.4f}")

        # Batch predictions to avoid query length limitations (1000 per batch)
        batch_size = 1000
        all_predictions = []

        # Get transaction_id (primary key) from test set
        test_trans_ids = test_transactions['transaction_id'].values
        num_batches = (len(test_trans_ids) + batch_size - 1) // batch_size

        print(f"\nProcessing {len(test_trans_ids)} transactions in {num_batches} batches of {batch_size}...")

        for i in range(0, len(test_trans_ids), batch_size):
            batch_num = i // batch_size + 1
            batch_ids = test_trans_ids[i:i + batch_size].tolist()

            # Create PQL query for this batch
            query = f"PREDICT transactions.is_fraud FOR transactions.transaction_id IN {tuple(batch_ids)}"

            print(f"\nBatch {batch_num}/{num_batches}: {len(batch_ids)} transactions (query length: {len(query)} chars)")

            try:
                # Make predictions for this batch
                batch_predictions = self.model.predict(query)
                all_predictions.append(batch_predictions)
                print(f"✓ Batch {batch_num} completed: {len(batch_predictions)} predictions")
            except Exception as e:
                print(f"✗ Error in batch {batch_num}: {str(e)}")
                raise

        # Combine all batch predictions
        predictions_df = pd.concat(all_predictions, ignore_index=True)

        # Store test transactions for evaluation
        self.test_transactions = test_transactions

        print(f"\n✓ All predictions completed: {len(predictions_df)} total predictions")
        print(f"Predictions shape: {predictions_df.shape}")

        return predictions_df

    def evaluate_predictions(self, predictions_df):
        """Evaluate model predictions"""
        print("\n=== KumoRFM Model Evaluation ===")

        # Use stored test transactions
        transactions_df = self.test_transactions

        # Merge predictions with actual labels
        results_df = transactions_df[['transaction_id', 'is_fraud']].merge(
            predictions_df,
            left_on='transaction_id',
            right_on='ENTITY',
            how='inner'
        )

        print(f"Evaluation set size: {len(results_df)}")
        print(f"Number of fraud cases: {results_df['is_fraud'].sum()}")
        print(f"Fraud rate: {results_df['is_fraud'].mean():.4f}")

        # Extract predictions
        y_true = results_df['is_fraud'].values

        # For binary classification, get the probability of fraud (class 1)
        if 'True_PROB' in predictions_df.columns:
            y_pred_proba = results_df['True_PROB'].values
        elif 'TARGET_PRED' in predictions_df.columns:
            y_pred_proba = results_df['TARGET_PRED'].values
        else:
            # Use the prediction column
            pred_col = [c for c in predictions_df.columns if 'PRED' in c][0]
            y_pred_proba = results_df[pred_col].values

        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        # Only calculate metrics if there are positive samples
        if y_true.sum() > 0:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            print(f"\nROC AUC Score: {roc_auc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            self.results = {
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        else:
            print("\n⚠ No positive samples in test set - cannot calculate meaningful metrics")
            self.results = {
                'roc_auc': np.nan,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

        return self.results

    def save_results(self, filepath='results/kumo_results.csv'):
        """Save results to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        results_df = pd.DataFrame([self.results])
        results_df.to_csv(filepath, index=False)
        print(f"\nResults saved to {filepath}")

    def run_full_pipeline(self):
        """Run the complete KumoRFM fraud detection pipeline"""
        print("=" * 80)
        print("KUMORFM FRAUD DETECTION PIPELINE")
        print("=" * 80)

        try:
            # Initialize KumoRFM
            self.initialize_kumo()

            # Load and prepare data
            transactions_df, cards_df, users_df = self.load_and_prepare_data()

            # Create graph
            self.create_graph(transactions_df, cards_df, users_df)

            # Train and predict (will use test set IDs from XGBoost)
            predictions_df = self.train_and_predict(transactions_df)

            # Evaluate on test set
            self.evaluate_predictions(predictions_df)

            # Save results
            self.save_results()

            print("\n=== KumoRFM Pipeline Completed Successfully ===")

        except Exception as e:
            print(f"\nError in KumoRFM pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nThis may be due to:")
            print("1. Invalid API key")
            print("2. Network connectivity issues")
            print("3. Data format issues")
            print("4. API quota exceeded")
            raise

        return self.results


def main():
    """Main execution function"""
    detector = KumoRFMFraudDetector()
    results = detector.run_full_pipeline()
    print("\n=== Final Results ===")
    print(results)


if __name__ == "__main__":
    main()
