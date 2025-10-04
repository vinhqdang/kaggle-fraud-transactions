"""
Configuration file for the fraud detection project.
Stores API keys and other configuration parameters.
"""

# Kumo AI API Configuration
KUMO_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1M2E3YTA2OGVjZDFlN2EzMDY5Y2E4MGYxNzU4MWU1YyIsImp0aSI6IjcwYWQ0OWFjLTIwYmYtNDBjZS04NTE2LTY4Mzg2NGRmMDFkMyIsImlhdCI6MTc1OTU0MDQ3NCwiZXhwIjoxNzY0NzI0NDc0fQ.WcNMYZsPfIXrwrwxT_nmpSa7NEYtoAPJA806a02QlAM"
KUMO_API_URL = "https://api.kumo.ai/api"  # Update this with the actual Kumo API URL

# Data paths
DATA_DIR = "data/"
TRANSACTIONS_FILE = "transactions_data.csv"
CARDS_FILE = "cards_data.csv"
USERS_FILE = "users_data.csv"
MCC_CODES_FILE = "mcc_codes.json"
FRAUD_LABELS_FILE = "train_fraud_labels.json"

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
