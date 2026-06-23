import pickle
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "artifacts" / "credit_score_model.pkl"
METRICS_PATH = BASE_DIR / "artifacts" / "model_metrics.json"

SCORE_LABELS = {0: "Poor", 1: "Standard", 2: "Good"}
SCORE_ORDER = ["Poor", "Standard", "Good"]
SCORE_COLORS = ["#ef4444", "#f59e0b", "#22c55e"]
MODEL_DISPLAY_NAMES = {
    "HistGradientBoostingClassifier": "Gradient Boosting",
    "LGBMClassifier": "LightGBM",
}

PROFILE_PRESETS = {
    "Balanced": {
        "Annual_Income": 50000,
        "Monthly_Inhand_Salary": 4000,
        "Outstanding_Debt": 5000,
        "Credit_Utilization_Ratio": 30,
        "Num_of_Delayed_Payment": 0,
        "Num_of_Loan": 2,
        "Num_Credit_Card": 2,
        "Interest_Rate": 10.0,
        "Num_Credit_Inquiries": 2,
        "Total_EMI_per_month": 500,
        "Age": 35,
        "Credit_History_Age": 60,
        "Changed_Credit_Limit": 10000,
        "Delay_from_due_date": 0,
        "Amount_invested_monthly": 500,
        "Monthly_Balance": 1000,
        "Num_Bank_Accounts": 2,
        "Credit_Mix": "Standard",
        "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Occupation": "Engineer",
    },
    "Strong Credit": {
        "Annual_Income": 95000,
        "Monthly_Inhand_Salary": 7200,
        "Outstanding_Debt": 1200,
        "Credit_Utilization_Ratio": 18,
        "Num_of_Delayed_Payment": 0,
        "Num_of_Loan": 1,
        "Num_Credit_Card": 4,
        "Interest_Rate": 7.5,
        "Num_Credit_Inquiries": 1,
        "Total_EMI_per_month": 250,
        "Age": 42,
        "Credit_History_Age": 180,
        "Changed_Credit_Limit": 18000,
        "Delay_from_due_date": -2,
        "Amount_invested_monthly": 1200,
        "Monthly_Balance": 2800,
        "Num_Bank_Accounts": 3,
        "Credit_Mix": "Good",
        "Payment_of_Min_Amount": "No",
        "Payment_Behaviour": "High_spent_Large_value_payments",
        "Occupation": "Manager",
    },
    "Needs Attention": {
        "Annual_Income": 32000,
        "Monthly_Inhand_Salary": 2400,
        "Outstanding_Debt": 18000,
        "Credit_Utilization_Ratio": 82,
        "Num_of_Delayed_Payment": 14,
        "Num_of_Loan": 7,
        "Num_Credit_Card": 6,
        "Interest_Rate": 24.0,
        "Num_Credit_Inquiries": 8,
        "Total_EMI_per_month": 1300,
        "Age": 29,
        "Credit_History_Age": 24,
        "Changed_Credit_Limit": 2500,
        "Delay_from_due_date": 28,
        "Amount_invested_monthly": 80,
        "Monthly_Balance": 120,
        "Num_Bank_Accounts": 6,
        "Credit_Mix": "Bad",
        "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Occupation": "Other",
    },
}

NUMERIC_CONFIG = {
    "Annual_Income": {"label": "Annual income ($)", "min": 0, "max": 500000, "step": 1000},
    "Monthly_Inhand_Salary": {"label": "Monthly in-hand salary ($)", "min": 0, "max": 50000, "step": 100},
    "Outstanding_Debt": {"label": "Outstanding debt ($)", "min": 0, "max": 100000, "step": 100},
    "Credit_Utilization_Ratio": {"label": "Credit utilization (%)", "min": 0, "max": 100, "step": 1},
    "Num_of_Delayed_Payment": {"label": "Delayed payments", "min": 0, "max": 50, "step": 1},
    "Num_of_Loan": {"label": "Loans", "min": 0, "max": 20, "step": 1},
    "Num_Credit_Card": {"label": "Credit cards", "min": 0, "max": 20, "step": 1},
    "Interest_Rate": {"label": "Interest rate (%)", "min": 0.0, "max": 30.0, "step": 0.5},
    "Num_Credit_Inquiries": {"label": "Credit inquiries", "min": 0, "max": 20, "step": 1},
    "Total_EMI_per_month": {"label": "Total EMI per month ($)", "min": 0, "max": 5000, "step": 50},
    "Age": {"label": "Age", "min": 18, "max": 90, "step": 1},
    "Credit_History_Age": {"label": "Credit history age (months)", "min": 0, "max": 600, "step": 6},
    "Changed_Credit_Limit": {"label": "Changed credit limit ($)", "min": 0, "max": 50000, "step": 500},
    "Delay_from_due_date": {"label": "Delay from due date (days)", "min": -30, "max": 90, "step": 1},
    "Amount_invested_monthly": {"label": "Amount invested monthly ($)", "min": 0, "max": 5000, "step": 50},
    "Monthly_Balance": {"label": "Monthly balance ($)", "min": 0, "max": 10000, "step": 100},
    "Num_Bank_Accounts": {"label": "Bank accounts", "min": 0, "max": 10, "step": 1},
}

CATEGORICAL_CONFIG = {
    "Credit_Mix": {
        "label": "Credit mix",
        "options": ["Bad", "Standard", "Good"],
    },
    "Payment_of_Min_Amount": {
        "label": "Payment of minimum amount",
        "options": ["No", "Yes", "NM"],
    },
    "Payment_Behaviour": {
        "label": "Payment behaviour",
        "options": [
            "Low_spent_Small_value_payments",
            "High_spent_Medium_value_payments",
            "Low_spent_Medium_value_payments",
            "High_spent_Large_value_payments",
            "High_spent_Small_value_payments",
            "Low_spent_Large_value_payments",
            "Unknown",
        ],
    },
    "Occupation": {
        "label": "Occupation",
        "options": [
            "Scientist",
            "Teacher",
            "Engineer",
            "Entrepreneur",
            "Doctor",
            "Lawyer",
            "Manager",
            "Accountant",
            "Musician",
            "Mechanic",
            "Writer",
            "Architect",
            "Developer",
            "Journalist",
            "Designer",
            "Other",
            "Unknown",
        ],
    },
}


st.set_page_config(
    page_title="Credit Score Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        line-height: 1.45;
    }
    .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        background: rgba(128, 128, 128, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None, (
            "Model artifact not found. Run "
            "`python3 train_model.py --train-csv data/raw/train.csv "
            "--model-out artifacts/credit_score_model.pkl` from the project root."
        )

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    if hasattr(model, "feature_name_"):
        feature_names = list(model.feature_name_)
    elif hasattr(model, "booster_"):
        feature_names = list(model.booster_.feature_name())
    elif hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = None

    return model, feature_names, None


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    return pd.read_json(METRICS_PATH, typ="series").to_dict()


def display_model_name(model):
    return MODEL_DISPLAY_NAMES.get(type(model).__name__, type(model).__name__)


def build_input_frame(form_data, feature_names):
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)

    for feature, value in form_data.items():
        if feature in input_df.columns:
            input_df[feature] = value

    credit_mix_map = {"Bad": 0, "Standard": 1, "Good": 2}
    if "Credit_Mix" in input_df.columns:
        input_df["Credit_Mix"] = credit_mix_map.get(form_data["Credit_Mix"], 1)

    one_hot_values = {
        "Payment_of_Min_Amount": form_data["Payment_of_Min_Amount"],
        "Payment_Behaviour": form_data["Payment_Behaviour"],
        "Occupation": form_data["Occupation"],
    }
    for prefix, value in one_hot_values.items():
        column = f"{prefix}_{value}"
        if column in input_df.columns:
            input_df[column] = 1

    ratio_values = {
        "debt_ratio": form_data["Outstanding_Debt"] / (form_data["Annual_Income"] + 1),
        "monthly_liabilities_ratio": form_data["Total_EMI_per_month"]
        / (form_data["Monthly_Inhand_Salary"] + 1),
        "loan_to_income_ratio": form_data["Num_of_Loan"] / (form_data["Annual_Income"] + 1),
        "salary_to_EMI_ratio": form_data["Monthly_Inhand_Salary"]
        / (form_data["Total_EMI_per_month"] + 1),
        "monthly_saving_ratio": form_data["Monthly_Balance"]
        / (form_data["Monthly_Inhand_Salary"] + 1),
        "debt_to_credit_ratio": form_data["Outstanding_Debt"]
        / (form_data["Changed_Credit_Limit"] + 1),
    }
    for column, value in ratio_values.items():
        if column in input_df.columns:
            input_df[column] = value

    return input_df


def predict_credit_score(form_data, model, feature_names):
    input_df = build_input_frame(form_data, feature_names)
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    class_labels = [SCORE_LABELS.get(int(class_value), str(class_value)) for class_value in model.classes_]
    probability_df = pd.DataFrame({"Category": class_labels, "Probability": probabilities})
    probability_df = probability_df.set_index("Category").reindex(SCORE_ORDER).fillna(0)

    return SCORE_LABELS.get(int(prediction), "Unknown"), probability_df


def probability_chart(probability_df):
    chart_df = probability_df.reset_index()
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Category:N", sort=SCORE_ORDER, title="Credit score category"),
            y=alt.Y(
                "Probability:Q",
                title="Model confidence",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(domain=SCORE_ORDER, range=SCORE_COLORS),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Category:N", title="Category"),
                alt.Tooltip("Probability:Q", title="Confidence", format=".1%"),
            ],
        )
        .properties(height=280, width="container")
    )
    return chart


def profile_signals(form_data):
    debt_ratio = form_data["Outstanding_Debt"] / (form_data["Annual_Income"] + 1)
    emi_ratio = form_data["Total_EMI_per_month"] / (form_data["Monthly_Inhand_Salary"] + 1)
    utilization = form_data["Credit_Utilization_Ratio"]
    delayed = form_data["Num_of_Delayed_Payment"]

    signals = []
    signals.append(("Debt burden", debt_ratio, "Outstanding debt relative to annual income"))
    signals.append(("Monthly EMI load", emi_ratio, "Monthly obligations relative to take-home pay"))
    signals.append(("Credit utilization", utilization / 100, "Share of available credit currently used"))
    signals.append(("Delayed payments", delayed / 50, "Recent payment delays"))
    return signals


model, feature_names, load_error = load_model()
metrics = load_metrics()

st.sidebar.title("How To Use")
st.sidebar.markdown(
    """
    1. Pick a sample profile.
    2. Adjust the financial and credit behavior inputs.
    3. Switch between the input tabs to edit more details.
    4. Read the predicted category and confidence chart.
    """
)
st.sidebar.divider()
st.sidebar.markdown(
    """
    **Sample profiles**

    **Balanced** starts with moderate income, debt, and utilization.

    **Strong Credit** shows a healthier payment and debt profile.

    **Needs Attention** shows higher utilization, debt, and delayed payments.
    """
)
st.sidebar.divider()
st.sidebar.info(
    "This is an educational portfolio demo. It should not be used for real credit decisions."
)

st.title("Credit Score Classification Demo")
st.caption(
    "Explore how income, debt, payment behavior, and credit history can influence "
    "a model's estimated credit-score category."
)

if load_error:
    st.warning(load_error)
    st.stop()

if model is None or feature_names is None:
    st.warning("The model could not be loaded with feature names.")
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Model", display_model_name(model))
metric_cols[1].metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
metric_cols[2].metric("Weighted F1", f"{metrics.get('weighted_f1', 0) * 100:.1f}%")
metric_cols[3].metric("Features", f"{len(feature_names)}")

st.info(
    "This demo predicts one of three categories: Poor, Standard, or Good. "
    "The result is based on patterns learned from the training data, not on a real credit bureau score."
)

with st.expander("What does the model look at?"):
    info_cols = st.columns(3)
    info_cols[0].markdown("**Ability to pay**\n\nIncome, monthly salary, EMI load, and monthly balance.")
    info_cols[1].markdown("**Credit behavior**\n\nUtilization, delayed payments, credit mix, and inquiries.")
    info_cols[2].markdown("**Credit history**\n\nCredit history age, loans, debt, and account counts.")

st.divider()

left_col, right_col = st.columns([1.15, 1])

with left_col:
    st.subheader("Customer Profile")
    profile_name = st.radio(
        "Sample profile",
        list(PROFILE_PRESETS.keys()),
        horizontal=True,
    )
    preset = PROFILE_PRESETS[profile_name]

    form_data = {}
    input_tabs = st.tabs(["Financials", "Credit Behavior", "Profile"])

    with input_tabs[0]:
        for feature in [
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Outstanding_Debt",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance",
        ]:
            config = NUMERIC_CONFIG[feature]
            form_data[feature] = st.number_input(
                config["label"],
                min_value=config["min"],
                max_value=config["max"],
                value=preset[feature],
                step=config["step"],
                key=f"{profile_name}_{feature}",
            )

    with input_tabs[1]:
        for feature in [
            "Credit_Utilization_Ratio",
            "Num_of_Delayed_Payment",
            "Num_of_Loan",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_Credit_Inquiries",
            "Changed_Credit_Limit",
            "Delay_from_due_date",
        ]:
            config = NUMERIC_CONFIG[feature]
            form_data[feature] = st.number_input(
                config["label"],
                min_value=config["min"],
                max_value=config["max"],
                value=preset[feature],
                step=config["step"],
                key=f"{profile_name}_{feature}",
            )

    with input_tabs[2]:
        for feature in ["Age", "Credit_History_Age", "Num_Bank_Accounts"]:
            config = NUMERIC_CONFIG[feature]
            form_data[feature] = st.number_input(
                config["label"],
                min_value=config["min"],
                max_value=config["max"],
                value=preset[feature],
                step=config["step"],
                key=f"{profile_name}_{feature}",
            )

        for feature, config in CATEGORICAL_CONFIG.items():
            form_data[feature] = st.selectbox(
                config["label"],
                options=config["options"],
                index=config["options"].index(preset[feature]),
                key=f"{profile_name}_{feature}",
            )

with right_col:
    predicted_label, probability_df = predict_credit_score(form_data, model, feature_names)
    confidence = probability_df.loc[predicted_label, "Probability"]

    st.subheader("Prediction")
    if predicted_label == "Good":
        st.success(f"Predicted category: {predicted_label}")
    elif predicted_label == "Standard":
        st.warning(f"Predicted category: {predicted_label}")
    else:
        st.error(f"Predicted category: {predicted_label}")

    st.metric("Model confidence", f"{confidence * 100:.1f}%")
    st.altair_chart(probability_chart(probability_df))

    prob_cols = st.columns(3)
    for col, (label, row) in zip(prob_cols, probability_df.iterrows()):
        col.metric(label, f"{row['Probability'] * 100:.1f}%")

    st.subheader("Risk Indicators")
    st.caption("Higher bars indicate more pressure in that area.")
    for label, value, help_text in profile_signals(form_data):
        st.progress(min(max(value, 0), 1), text=f"{label}: {help_text}")

st.divider()
st.caption(
    "Educational demo only. Credit scoring is a high-impact financial use case; "
    "real systems require fairness testing, explainability, monitoring, and compliance review."
)
