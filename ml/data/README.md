# Data Directory

This directory is reserved for local data artifacts used by the Financial Fraud Detection System.

## Source Dataset

The project uses:

**CiferAI/Cifer-Fraud-Detection-Dataset-AF**

The dataset is accessed programmatically through the Hugging Face `datasets` library.

The original dataset contains synthetic financial transactions generated for fraud-detection research and includes transaction attributes such as:

- transaction type
- transaction amount
- sender balance before the transaction
- sender balance after the transaction
- recipient balance before the transaction
- recipient balance after the transaction
- fraud label

## Model Features

The fraud detector uses the following fields:

- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

The supervised target is:

- `isFraud`

The label mapping used by the project is:

- `isFraud = 1` → `HIGH`
- `isFraud = 0` → `LOW`

## Data Pipeline

The project processes the source dataset using the following pipeline:

```text
Cifer source dataset
        |
        v
Reproducible working subset
        |
        v
Schema validation
        |
        v
Transaction preprocessing
        |
        v
Independent train/test split
        |
        +----------------------+
        |                      |
        v                      v
Training partition       Test partition
        |                      |
        v                      v
Class balancing          Class balancing
        |                      |
        v                      v
Fine-tuning data         Unseen evaluation data