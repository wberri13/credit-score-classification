"""Train the model used by the Streamlit credit score demo."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


TARGET_MAP = {"Poor": 0, "Standard": 1, "Good": 2}

NUMERIC_FEATURES = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Num_of_Delayed_Payment",
    "Num_of_Loan",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_Credit_Inquiries",
    "Total_EMI_per_month",
    "Age",
    "Credit_History_Age",
    "Changed_Credit_Limit",
    "Delay_from_due_date",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Num_Bank_Accounts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        default="data/raw/train.csv",
        help="Path to the labeled training CSV.",
    )
    parser.add_argument(
        "--model-out",
        default="artifacts/credit_score_model.pkl",
        help="Path where the trained model is written.",
    )
    parser.add_argument(
        "--metrics-out",
        default="artifacts/model_metrics.json",
        help="Path where metrics are written.",
    )
    parser.add_argument(
        "--model-type",
        choices=["auto", "lightgbm", "hist-gradient-boosting"],
        default="auto",
        help="Model family to train. Auto uses LightGBM when it can load, otherwise scikit-learn.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row sample for quick local smoke tests.",
    )
    return parser.parse_args()


def build_model(model_type: str) -> tuple[Any, str]:
    if model_type in {"auto", "lightgbm"}:
        try:
            from lightgbm import LGBMClassifier

            return (
                LGBMClassifier(
                    objective="multiclass",
                    num_class=3,
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=67,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
                "LightGBM",
            )
        except (ImportError, OSError) as exc:
            if model_type == "lightgbm":
                raise RuntimeError(
                    "LightGBM could not load. On macOS, install libomp with "
                    "`brew install libomp`, or rerun with "
                    "`--model-type hist-gradient-boosting`."
                ) from exc
            print(f"LightGBM unavailable ({exc}); falling back to scikit-learn.")

    return (
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            random_state=67,
        ),
        "HistGradientBoostingClassifier",
    )


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace("_", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def credit_history_to_months(value: object) -> float:
    if pd.isna(value):
        return np.nan
    value_text = str(value)
    if value_text.upper() == "NA":
        return np.nan
    parts = value_text.split()
    if len(parts) < 4:
        return np.nan
    try:
        return float(parts[0]) * 12 + float(parts[3])
    except ValueError:
        return np.nan


def cap_outliers(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        lower = frame[column].quantile(0.005)
        upper = frame[column].quantile(0.995)
        frame[column] = frame[column].clip(lower=lower, upper=upper)
    return frame


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = raw.copy()

    if "Credit_Score" not in df.columns:
        raise ValueError("Expected a labeled training file with a Credit_Score column.")

    df["Credit_History_Age"] = df["Credit_History_Age"].apply(credit_history_to_months)

    for column in NUMERIC_FEATURES:
        df[column] = clean_numeric(df[column])

    non_negative_columns = [
        column for column in NUMERIC_FEATURES if column != "Delay_from_due_date"
    ]
    for column in non_negative_columns:
        df.loc[df[column] < 0, column] = np.nan

    df.loc[(df["Age"] < 18) | (df["Age"] > 100), "Age"] = np.nan

    for column in NUMERIC_FEATURES:
        df[column] = df[column].fillna(df[column].median())

    df = cap_outliers(df, NUMERIC_FEATURES)

    df["Credit_Mix"] = df["Credit_Mix"].replace({"_": "Standard"}).fillna("Standard")
    df["Credit_Mix"] = (
        df["Credit_Mix"].map({"Bad": 0, "Standard": 1, "Good": 2}).fillna(1)
    )

    df["Payment_Behaviour"] = (
        df["Payment_Behaviour"].replace({"!@9#%8": "Unknown"}).fillna("Unknown")
    )
    df["Occupation"] = df["Occupation"].replace({"_______": "Unknown"}).fillna("Unknown")
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].fillna("NM")

    df["debt_ratio"] = df["Outstanding_Debt"] / (df["Annual_Income"] + 1)
    df["monthly_liabilities_ratio"] = df["Total_EMI_per_month"] / (
        df["Monthly_Inhand_Salary"] + 1
    )
    df["loan_to_income_ratio"] = df["Num_of_Loan"] / (df["Annual_Income"] + 1)
    df["salary_to_EMI_ratio"] = df["Monthly_Inhand_Salary"] / (
        df["Total_EMI_per_month"] + 1
    )
    df["monthly_saving_ratio"] = df["Monthly_Balance"] / (
        df["Monthly_Inhand_Salary"] + 1
    )
    df["debt_to_credit_ratio"] = df["Outstanding_Debt"] / (
        df["Changed_Credit_Limit"] + 1
    )

    feature_frame = df[NUMERIC_FEATURES + ["Credit_Mix"]].copy()
    encoded = pd.get_dummies(
        df[["Payment_of_Min_Amount", "Payment_Behaviour", "Occupation"]],
        columns=["Payment_of_Min_Amount", "Payment_Behaviour", "Occupation"],
        dtype=int,
    )
    feature_frame = pd.concat([feature_frame, encoded], axis=1)

    engineered = [
        "debt_ratio",
        "monthly_liabilities_ratio",
        "loan_to_income_ratio",
        "salary_to_EMI_ratio",
        "monthly_saving_ratio",
        "debt_to_credit_ratio",
    ]
    feature_frame = pd.concat([feature_frame, df[engineered]], axis=1)
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0)

    target = df["Credit_Score"].map(TARGET_MAP)
    valid_rows = target.notna()
    return feature_frame.loc[valid_rows], target.loc[valid_rows].astype(int)


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str,
) -> tuple[Any, str, dict[str, float]]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=67,
        stratify=target,
    )

    model, model_name = build_model(model_type)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "weighted_precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "weighted_recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "weighted_f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }
    return model, model_name, metrics


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train_csv)
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)

    raw = pd.read_csv(train_csv, low_memory=False)
    if args.sample_size:
        raw = raw.sample(n=min(args.sample_size, len(raw)), random_state=67)

    features, target = preprocess(raw)
    model, model_name, metrics = train_model(features, target, args.model_type)

    with model_out.open("wb") as model_file:
        pickle.dump(model, model_file)

    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Trained model: {model_name}")
    print(f"Saved model to {model_out}")
    print(f"Saved metrics to {metrics_out}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
