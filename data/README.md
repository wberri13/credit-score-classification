# Data

This folder contains a small anonymized sample for repository review and local
smoke tests.

The original project data included direct identifier-style columns such as
`Name`, `SSN`, `ID`, and `Customer_ID`. Those columns are not needed for
modeling and should not be published as portfolio data.

For full retraining, place the original training file at:

```text
data/raw/train.csv
```

Then run:

```bash
python3 train_model.py --train-csv data/raw/train.csv --model-out artifacts/credit_score_model.pkl
```
