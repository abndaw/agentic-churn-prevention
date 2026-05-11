"""
Churn Prevention App
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Path setup 
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent import RetentionAgent
from preprocess import preprocess


# Page config
st.set_page_config(
    page_title="Churn Prevention",
    page_icon="📉",
    layout="wide"
)

st.title("Churn Prevention System")
st.caption("ML + SHAP + LLM personalized retention demo")

# Loading model & data
@st.cache_resource
def load_agent():
    model_path = PROJECT_ROOT / "models" / "best_model.pkl"
    threshold_path = PROJECT_ROOT / "outputs" / "threshold_results.json"

    return RetentionAgent(
        model_path=str(model_path),
        api_key=os.getenv("GOOGLE_API_KEY"),
        threshold_path=str(threshold_path)
    )


@st.cache_data
def load_data():
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess(scale_numeric=False)
    return X_test, y_test


agent = load_agent()
X_test, y_test = load_data()

# FILTER TABLE
@st.cache_data
def build_customer_table(X, y):

    # ML probabilities
    churn_probas = agent.model.predict_proba(X)[:, 1]

    # Risk levels
    risk_levels = pd.cut(
        churn_probas,
        bins=[0, 0.30, 0.90, 1],
        labels=["Low", "Medium", "High"]
    )

    df = pd.DataFrame({
        "customer_id": X.index,
        "churn_probability": churn_probas,
        "risk_level": risk_levels.astype(str),
        "actual_churn": y.values
    })

    return df


customers_df = build_customer_table(X_test, y_test)

# SIDEBAR FILTERS
st.sidebar.header("Filters")

# Probability interval
min_proba, max_proba = st.sidebar.slider(
    "Churn probability interval",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.01
)

# Risk levels
risk_options = ["Low", "Medium", "High"]

selected_risks = st.sidebar.multiselect(
    "Risk levels",
    options=risk_options,
    default=risk_options
)

# Churn status
churn_filter = st.sidebar.radio(
    "Actual status",
    ["All", "Churners only", "Non-churners only"]
)

# APPLYING FILTERS
filtered_df = customers_df[
    (customers_df["churn_probability"] >= min_proba) &
    (customers_df["churn_probability"] <= max_proba) &
    (customers_df["risk_level"].isin(selected_risks))
]

if churn_filter == "Churners only":
    filtered_df = filtered_df[filtered_df["actual_churn"] == 1]

elif churn_filter == "Non-churners only":
    filtered_df = filtered_df[filtered_df["actual_churn"] == 0]

# Customer selection
st.sidebar.header("Customer selection")

available_customers = filtered_df["customer_id"].tolist()

if len(available_customers) == 0:
    st.warning("No customers match selected filters.")
    st.stop()

customer_id = st.sidebar.selectbox(
    "Choose customer",
    available_customers
)

if st.sidebar.button("Random customer"):
    customer_id = np.random.choice(available_customers)

customer = X_test.loc[[customer_id]]

# Analysis
result = agent.analyze(customer, customer_id=customer_id)
true_label = y_test.loc[customer_id]

# Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Churn probability",
        f"{result['churn_probability']:.2%}"
    )

with col2:
    st.metric(
        "Risk level",
        result["action"]["risk"]
    )

with col3:
    st.metric(
        "Actual",
        "Churn" if true_label == 1 else "No churn"
    )

st.divider()

# Risk factors
st.subheader("Risk factors")

if result.get("risk_factors"):
    df = pd.DataFrame(result["risk_factors"])
    st.dataframe(df, use_container_width=True)
else:
    st.info("No major risk factors detected")

st.divider()

# OFFERS
st.subheader("Offers")

offers = result.get("offers", [])

if offers:
    for o in offers:

        st.markdown(f"""
        ### {o['name']}

        {o.get('benefit', '')}

        💰 {o.get('discount', '')}
        """)

        st.write("---")

else:
    st.info("No offers for this customer")

st.divider()

# EMAIL
st.subheader("📧 Email")

email = result.get("email", {})

st.write("**Subject:**", email.get("subject", ""))

st.text_area(
    "Body",
    email.get("body", ""),
    height=250
)

# DEBUG
with st.expander("Debug"):
    st.json(result)