"""
Project-wide constants for the Financial Fraud Detection System.

This module centralizes dataset metadata, fraud labels, reproducibility
settings, sampling sizes, split configuration, and model identifiers.

Keeping these values in one location prevents hard-coded values from being
duplicated across data preparation, training, evaluation, and inference code.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# constants.py
# ml/src/utils/constants.py
#
# parents[0] -> utils
# parents[1] -> src
# parents[2] -> ml
ML_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = ML_ROOT.parent

DATA_DIR = ML_ROOT / "data"

CONFIG_DIR = ML_ROOT / "configs"

NOTEBOOK_DIR = ML_ROOT / "notebooks"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

EVALUATION_RESULTS_DIR = EVALUATION_DIR / "results"

EVALUATION_FIGURES_DIR = EVALUATION_DIR / "figures"


# =============================================================================
# DATASET
# =============================================================================

DATASET_ID = "CiferAI/Cifer-Fraud-Detection-Dataset-AF"

DATASET_SPLIT = "train"

WORKING_SUBSET_SIZE = 1_000_000


# =============================================================================
# REQUIRED DATASET COLUMNS
# =============================================================================

TRANSACTION_TYPE_COLUMN = "type"

AMOUNT_COLUMN = "amount"

SENDER_BALANCE_BEFORE_COLUMN = "oldbalanceOrg"

SENDER_BALANCE_AFTER_COLUMN = "newbalanceOrig"

RECIPIENT_BALANCE_BEFORE_COLUMN = "oldbalanceDest"

RECIPIENT_BALANCE_AFTER_COLUMN = "newbalanceDest"

FRAUD_LABEL_COLUMN = "isFraud"


FEATURE_COLUMNS = (
    TRANSACTION_TYPE_COLUMN,
    AMOUNT_COLUMN,
    SENDER_BALANCE_BEFORE_COLUMN,
    SENDER_BALANCE_AFTER_COLUMN,
    RECIPIENT_BALANCE_BEFORE_COLUMN,
    RECIPIENT_BALANCE_AFTER_COLUMN,
)

REQUIRED_COLUMNS = FEATURE_COLUMNS + (FRAUD_LABEL_COLUMN,)


# =============================================================================
# FRAUD LABELS
# =============================================================================

FRAUD_CLASS = 1

LEGITIMATE_CLASS = 0

HIGH_RISK_LABEL = "HIGH"

LOW_RISK_LABEL = "LOW"


CLASS_TO_RISK_LABEL = {
    FRAUD_CLASS: HIGH_RISK_LABEL,
    LEGITIMATE_CLASS: LOW_RISK_LABEL,
}

RISK_LABEL_TO_CLASS = {
    HIGH_RISK_LABEL: FRAUD_CLASS,
    LOW_RISK_LABEL: LEGITIMATE_CLASS,
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

RANDOM_SEED = 42


# =============================================================================
# TRAIN / TEST SPLITTING
# =============================================================================

TRAIN_SPLIT_RATIO = 0.80

TEST_SPLIT_RATIO = 0.20


# =============================================================================
# BALANCED DATASET SIZES
# =============================================================================

# Training dataset:
# 1,000 fraud + 1,000 legitimate = 2,000 examples.
#
# This is intentionally larger than the 1,000-example educational version
# while remaining practical for QLoRA experimentation on limited GPU resources.
TRAIN_SAMPLES_PER_CLASS = 1_000

TRAIN_DATASET_SIZE = TRAIN_SAMPLES_PER_CLASS * 2


# Evaluation dataset:
# 250 fraud + 250 legitimate = 500 unseen examples.
#
# The evaluation split is created before sampling, which prevents leakage
# between training and evaluation examples.
TEST_SAMPLES_PER_CLASS = 250

TEST_DATASET_SIZE = TEST_SAMPLES_PER_CLASS * 2


# =============================================================================
# BASE MODEL
# =============================================================================

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


# =============================================================================
# MODEL OUTPUT
# =============================================================================

VALID_RISK_LABELS = (
    HIGH_RISK_LABEL,
    LOW_RISK_LABEL,
)


# =============================================================================
# TRANSACTION TYPES
# =============================================================================

EXPECTED_TRANSACTION_TYPES = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)


# =============================================================================
# PROMPT
# =============================================================================

FRAUD_ANALYSIS_INSTRUCTION = "Analyze this transaction for fraud risk:"


# =============================================================================
# NUMERICAL VALIDATION
# =============================================================================

MIN_TRANSACTION_AMOUNT = 0.0

MIN_BALANCE = 0.0