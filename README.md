# American Express Credit Score Classification

End-to-end machine learning demo that classifies a customer profile as `Poor`,
`Standard`, or `Good` using income, debt, payment behavior, credit history, and
account-level financial indicators.

This project was built through Break Through Tech AI Studio with an American
Express challenge advisor. My main contributions were exploratory data
analysis, data cleaning, preprocessing decisions, and feature selection.

## Demo

Live app: [wafa-credit-score.streamlit.app](https://wafa-credit-score.streamlit.app/)

The Streamlit app is designed to run directly from this repository:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app includes:

- preset customer profiles for quick exploration
- editable financial and credit behavior inputs
- predicted credit score category
- confidence chart for all three classes
- model metrics and responsible-use note

## Portfolio Snapshot

- Built an end-to-end supervised learning workflow for multi-class credit score classification.
- Cleaned noisy tabular data with missing values, malformed numeric fields, categorical placeholders, and extreme outliers.
- Compared tree-based models including XGBoost, CatBoost, LightGBM, Decision Tree, and a neural-network baseline.
- Trained a deployable gradient-boosted classifier with approximately 74.5% accuracy and 74.5% weighted F1 on the project split.
- Packaged a Streamlit demo with a committed model artifact so the app works without retraining.
- Removed raw identifier-style columns from public sample data and documented responsible use for a high-impact finance use case.

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Streamlit
- Jupyter/Colab notebooks
- LightGBM, XGBoost, and CatBoost in notebook experiments

## Repository Structure

```text
.
|-- app.py                         # Streamlit demo
|-- train_model.py                 # Reproducible training script
|-- requirements.txt               # Runtime dependencies
|-- artifacts/
|   |-- credit_score_model.pkl      # Demo model artifact
|   `-- model_metrics.json          # Saved evaluation metrics
|-- data/
|   |-- README.md                   # Public data note
|   `-- sample_credit_scores.csv    # Sanitized sample rows for review
`-- notebooks/                     # Original EDA and modeling notebooks
```

## Quickstart

Clone the repository:

```bash
git clone https://github.com/wberri13/credit-score-classification.git
cd credit-score-classification
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

To deploy on Streamlit Community Cloud, connect this GitHub repository and set
the app entry point to:

```text
app.py
```

## Retraining

The public repository includes a small sanitized sample dataset, not the full
raw training file. For full retraining, place the original training CSV at:

```text
data/raw/train.csv
```

Then run:

```bash
python3 train_model.py --train-csv data/raw/train.csv --model-out artifacts/credit_score_model.pkl
```

The training script uses LightGBM when it can load. If LightGBM is unavailable,
it falls back to scikit-learn's histogram gradient boosting classifier. On
macOS, LightGBM may require OpenMP support:

```bash
pip install lightgbm
brew install libomp
```

## Modeling Approach

1. Removed direct identifiers from the modeling features.
2. Converted malformed numeric columns into usable numeric types.
3. Filled missing values using median or domain-appropriate defaults.
4. Converted credit history age into months.
5. Encoded ordinal and categorical variables.
6. Added ratio features for debt burden, monthly liabilities, loan-to-income, EMI coverage, savings, and debt-to-credit exposure.
7. Trained and evaluated a gradient-boosted classifier using an 80/20 stratified train/test split.

## Results

The deployed artifact was trained with the reproducible script and achieved:

| Metric | Score |
| --- | ---: |
| Accuracy | 74.5% |
| Weighted precision | 74.6% |
| Weighted recall | 74.5% |
| Weighted F1 | 74.5% |

The notebooks also document earlier experiments across XGBoost, CatBoost,
LightGBM, Decision Tree, and neural-network baselines.

## Data and Ethics Notes

Credit scoring is a high-impact financial use case. This project is an
educational classification demo, not a production underwriting system.

Important limitations:

- The public sample data excludes direct identifier-style columns such as names, SSNs, and customer IDs.
- Model outputs should not be used for real financial decisions without fairness testing, explainability review, monitoring, and compliance checks.
- Accuracy alone is not enough for a deployed credit model. Future work should include per-class error analysis, bias checks, calibration, and model explainability.

## Collaborators

- Wafa Berri
- Jason Lei
- Allison Romero
- Sheena Ansari
- Jerry Lin
- Kareem Khusenov
- Kashish Bhandari

## Acknowledgements

Thanks to Break Through Tech, American Express, AI Studio Coach Jenna Hunte, and
Challenge Advisor Saurabh Gupta for guidance and support throughout the project.
