"""
Kumo AI Fraud Detection Implementation
Using Kumo's relational foundation model for fraud detection
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
from datetime import datetime

try:
    import kumo
except ImportError:
    print("Warning: Kumo SDK not installed. Install with: pip install kumo")
    kumo = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)

from config import DATA_DIR, KUMO_API_KEY, KUMO_API_URL, RANDOM_SEED


class KumoFraudDetector:
    """Kumo AI-based fraud detection model"""

    def __init__(self, api_key=KUMO_API_KEY, api_url=KUMO_API_URL):
        self.api_key = api_key
        self.api_url = api_url
        self.connector = None
        self.graph = None
        self.model = None
        self.results = {}

    def initialize_kumo(self):
        """Initialize Kumo SDK"""
        if kumo is None:
            raise ImportError("Kumo SDK is not installed. Install with: pip install kumo")

        print("Initializing Kumo SDK...")
        kumo.init(url=self.api_url, api_key=self.api_key)
        print("Kumo SDK initialized successfully")

    def load_and_prepare_data(self):
        """Load and prepare data for Kumo"""
        print("Loading data files...")

        # Load all data files
        with open(os.path.join(DATA_DIR, 'train_fraud_labels.json'), 'r') as f:
            raw_json_data = json.load(f)
        transaction_labels_dict = raw_json_data['target']
        fraud_labels_df = pd.DataFrame([
            {'transaction_id': k, 'is_fraud': 1 if v == 'Yes' else 0}
            for k, v in transaction_labels_dict.items()
        ])
        fraud_labels_df['transaction_id'] = pd.to_numeric(fraud_labels_df['transaction_id'])

        transactions_df = pd.read_csv(os.path.join(DATA_DIR, 'transactions_data.csv'))
        cards_df = pd.read_csv(os.path.join(DATA_DIR, 'cards_data.csv'))
        users_df = pd.read_csv(os.path.join(DATA_DIR, 'users_data.csv'))

        # Load MCC codes
        with open(os.path.join(DATA_DIR, 'mcc_codes.json'), 'r') as f:
            mcc_data = json.load(f)
        mcc_df = pd.DataFrame([
            {'mcc_code': int(k), 'mcc_description': v}
            for k, v in mcc_data.items()
        ])

        # Merge fraud labels with transactions
        transactions_df = transactions_df.merge(
            fraud_labels_df,
            left_on='id',
            right_on='transaction_id',
            how='left'
        )

        print(f"Transactions shape: {transactions_df.shape}")
        print(f"Cards shape: {cards_df.shape}")
        print(f"Users shape: {users_df.shape}")
        print(f"MCC codes shape: {mcc_df.shape}")

        return transactions_df, cards_df, users_df, mcc_df

    def save_data_for_kumo(self, transactions_df, cards_df, users_df, mcc_df):
        """Save data in formats suitable for Kumo (CSV/Parquet)"""
        print("Saving data for Kumo...")

        kumo_data_dir = os.path.join(DATA_DIR, 'kumo_input')
        os.makedirs(kumo_data_dir, exist_ok=True)

        # Save as parquet for better performance
        transactions_df.to_parquet(os.path.join(kumo_data_dir, 'transactions.parquet'), index=False)
        cards_df.to_parquet(os.path.join(kumo_data_dir, 'cards.parquet'), index=False)
        users_df.to_parquet(os.path.join(kumo_data_dir, 'users.parquet'), index=False)
        mcc_df.to_parquet(os.path.join(kumo_data_dir, 'mcc_codes.parquet'), index=False)

        print(f"Data saved to {kumo_data_dir}")
        return kumo_data_dir

    def create_kumo_connector(self, kumo_data_dir):
        """Create Kumo file connector"""
        print("Creating Kumo connector...")

        # Use FileUploadConnector for local files
        self.connector = kumo.FileUploadConnector(root_dir=kumo_data_dir)
        print("Connector created successfully")

        return self.connector

    def build_graph(self):
        """Build Kumo graph from data"""
        print("Building Kumo graph...")

        # Create tables from source data
        transactions_table = kumo.Table.from_source_table(
            source_table=self.connector['transactions'],
            primary_key='id'
        ).infer_metadata()

        cards_table = kumo.Table.from_source_table(
            source_table=self.connector['cards'],
            primary_key='id'
        ).infer_metadata()

        users_table = kumo.Table.from_source_table(
            source_table=self.connector['users'],
            primary_key='id'
        ).infer_metadata()

        mcc_table = kumo.Table.from_source_table(
            source_table=self.connector['mcc_codes'],
            primary_key='mcc_code'
        ).infer_metadata()

        # Define graph with relationships
        self.graph = kumo.Graph(
            tables={
                'transactions': transactions_table,
                'cards': cards_table,
                'users': users_table,
                'mcc': mcc_table
            },
            edges=[
                # Transaction -> Card relationship
                dict(
                    src_table='transactions',
                    fkey='card_id',
                    dst_table='cards'
                ),
                # Transaction -> User relationship (via client_id)
                dict(
                    src_table='transactions',
                    fkey='client_id',
                    dst_table='users'
                ),
                # Transaction -> MCC relationship
                dict(
                    src_table='transactions',
                    fkey='mcc',
                    dst_table='mcc'
                )
            ]
        )

        print("Graph built successfully")
        return self.graph

    def create_predictive_query(self):
        """Create Kumo predictive query for fraud detection"""
        print("Creating predictive query...")

        # Define the prediction task
        pquery = kumo.PredictiveQuery(
            graph=self.graph,
            query="""
            PREDICT transactions.is_fraud
            FOR EACH transactions.id
            """
        )

        print("Predictive query created")
        return pquery

    def train_model(self, pquery):
        """Train Kumo model"""
        print("Training Kumo model...")

        # Train the model
        self.model = pquery.train()

        print("Model training completed")
        return self.model

    def make_predictions(self):
        """Make predictions using trained model"""
        print("Making predictions...")

        # Get predictions
        predictions = self.model.predict()

        print("Predictions completed")
        return predictions

    def evaluate_predictions(self, predictions, transactions_df):
        """Evaluate model predictions"""
        print("\n=== Kumo Model Evaluation ===")

        # Merge predictions with actual labels
        results_df = transactions_df[['id', 'is_fraud']].merge(
            predictions,
            left_on='id',
            right_on='transactions_id',
            how='inner'
        )

        # Remove rows with missing fraud labels (validation only on labeled data)
        results_df = results_df.dropna(subset=['is_fraud'])

        y_true = results_df['is_fraud'].values
        y_pred_proba = results_df['prediction'].values
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        roc_auc = roc_auc_score(y_true, y_pred_proba)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

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

        return self.results

    def save_results(self, filepath='results/kumo_results.csv'):
        """Save results to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        results_df = pd.DataFrame([self.results])
        results_df.to_csv(filepath, index=False)
        print(f"\nResults saved to {filepath}")

    def run_full_pipeline(self):
        """Run the complete Kumo fraud detection pipeline"""
        print("=" * 80)
        print("KUMO AI FRAUD DETECTION PIPELINE")
        print("=" * 80)

        try:
            # Initialize Kumo
            self.initialize_kumo()

            # Load and prepare data
            transactions_df, cards_df, users_df, mcc_df = self.load_and_prepare_data()

            # Save data for Kumo
            kumo_data_dir = self.save_data_for_kumo(transactions_df, cards_df, users_df, mcc_df)

            # Create connector
            self.create_kumo_connector(kumo_data_dir)

            # Build graph
            self.build_graph()

            # Create predictive query
            pquery = self.create_predictive_query()

            # Train model
            self.train_model(pquery)

            # Make predictions
            predictions = self.make_predictions()

            # Evaluate
            self.evaluate_predictions(predictions, transactions_df)

            # Save results
            self.save_results()

            print("\n=== Kumo Pipeline Completed Successfully ===")

        except Exception as e:
            print(f"\nError in Kumo pipeline: {str(e)}")
            print("This may be due to:")
            print("1. Kumo SDK not installed (pip install kumo)")
            print("2. Invalid API key or URL")
            print("3. Network connectivity issues")
            print("4. Data format issues")
            raise

        return self.results


def main():
    """Main execution function"""
    detector = KumoFraudDetector()
    results = detector.run_full_pipeline()
    print("\n=== Final Results ===")
    print(results)


if __name__ == "__main__":
    main()
