# import streamlit as st
# import pandas as pd
# import requests
# import json

# st.set_page_config(page_title="FraudOps Dashboard", layout="wide")
# st.title("🛡️ FraudOps — Transaction Monitoring")

# API_URL = "http://127.0.0.1:8000/predict"

# # Load a sample of transactions to simulate a "live feed"
# @st.cache_data
# def load_data():
#     df = pd.read_csv("creditcard.csv")
#     df = df.sort_values("Time").reset_index(drop=True)
#     n = len(df)
#     val_end = int(n * 0.85)
#     test_df = df.iloc[val_end:]  # same test split logic as train.py
#     return test_df.sample(50, random_state=1).reset_index(drop=True)

# df = load_data()
# st.write(f"Debug — showing {len(df)} transactions, Time range: {df['Time'].min():.0f} to {df['Time'].max():.0f}")
# feature_cols = [c for c in df.columns if c != "Class"]

# if "reviewed" not in st.session_state:
#     st.session_state.reviewed = {}

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.subheader("Recent Transactions")
#     for idx, row in df.iterrows():
#         payload = row[feature_cols].to_dict()
#         response = requests.post(API_URL, json=payload).json()

#         flagged = response["is_fraud"]
#         prob = response["fraud_probability"]

#         with st.expander(
#             f"{'🚨' if flagged else '✅'} Transaction #{idx} — "
#             f"Fraud Probability: {prob:.2%}"
#         ):
#             st.write(f"**Amount:** ${row['Amount']:.2f}")
#             st.write(f"**Actual label (ground truth):** "
#                      f"{'Fraud' if row['Class']==1 else 'Legit'}")
#             st.write("**Top contributing features:**")
#             for feat in response["top_contributing_features"]:
#                 st.write(f"- {feat['feature']}: {feat['impact']:+.3f}")

#             review_key = f"review_{idx}"
#             if st.button(f"Mark as reviewed", key=review_key):
#                 st.session_state.reviewed[idx] = True

#             if st.session_state.reviewed.get(idx):
#                 st.success("✔️ Reviewed by analyst")

# with col2:
#     st.subheader("Model Performance")
#     st.metric("Threshold Used", f"{response['threshold_used']:.4f}")
#     st.metric("Transactions Flagged", int(df.apply(
#         lambda r: requests.post(API_URL, json=r[feature_cols].to_dict()).json()["is_fraud"], axis=1
#     ).sum()) if False else "—")
#     st.caption("Precision: 63% | Recall: 75% | PR-AUC: 0.78")
#     st.caption("(from held-out test set evaluation)")
#     st.metric("Transactions Reviewed", len(st.session_state.reviewed))


import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="FraudOps Dashboard", layout="wide")
st.title("🛡️ FraudOps — Transaction Monitoring")

API_URL = "http://127.0.0.1:8000/predict"

@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    val_end = int(n * 0.85)
    test_df = df.iloc[val_end:]  # test set only — same split logic as train.py
    return test_df.sample(50, random_state=1).reset_index(drop=True)

df = load_data()
feature_cols = [c for c in df.columns if c != "Class"]

if "reviewed" not in st.session_state:
    st.session_state.reviewed = {}

# ---------------------------------------------------------
# SECTION 1: Manual transaction tester (moved to top)
# ---------------------------------------------------------
st.subheader("🔍 Test a Custom Transaction")

with st.form("manual_test"):
    amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
    time_val = st.number_input("Time (seconds since first transaction)", value=50000.0)
    v14 = st.slider("V14", -20.0, 10.0, 0.0)
    v17 = st.slider("V17", -15.0, 10.0, 0.0)

    submitted = st.form_submit_button("Check Transaction")

    if submitted:
        baseline = df[df["Class"] == 0][feature_cols].mean().to_dict()
        baseline["Amount"] = amount
        baseline["Time"] = time_val
        baseline["V14"] = v14
        baseline["V17"] = v17

        result = requests.post(API_URL, json=baseline).json()

        if result["is_fraud"]:
            st.error(f"🚨 FLAGGED AS FRAUD — Probability: {result['fraud_probability']:.2%}")
        else:
            st.success(f"✅ Looks legitimate — Probability: {result['fraud_probability']:.2%}")

        st.write("**Top contributing features:**")
        for feat in result["top_contributing_features"]:
            st.write(f"- {feat['feature']}: {feat['impact']:+.3f}")

st.divider()

# ---------------------------------------------------------
# SECTION 2: Historical test-set transactions + model stats
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Recent Transactions (held-out test set)")
    results = []
    for idx, row in df.iterrows():
        payload = row[feature_cols].to_dict()
        response = requests.post(API_URL, json=payload).json()
        flagged = response["is_fraud"]
        prob = response["fraud_probability"]
        results.append({"idx": idx, "predicted": flagged, "actual": row["Class"] == 1})

        with st.expander(
            f"{'🚨' if flagged else '✅'} Transaction #{idx} — Fraud Probability: {prob:.2%}"
        ):
            st.write(f"**Amount:** ${row['Amount']:.2f}")
            st.write(f"**Actual label (ground truth):** {'Fraud' if row['Class']==1 else 'Legit'}")
            st.write("**Top contributing features:**")
            for feat in response["top_contributing_features"]:
                st.write(f"- {feat['feature']}: {feat['impact']:+.3f}")

            review_key = f"review_{idx}"
            if st.button("Mark as reviewed", key=review_key):
                st.session_state.reviewed[idx] = True
            if st.session_state.reviewed.get(idx):
                st.success("✔️ Reviewed by analyst")

with col2:
    st.subheader("Model Performance")
    results_df = pd.DataFrame(results)
    n_flagged = int(results_df["predicted"].sum())
    n_correct = int((results_df["predicted"] == results_df["actual"]).sum())

    st.metric("Threshold Used", "0.3777")
    st.metric("Transactions Flagged", n_flagged)
    st.metric("Correctly Classified", f"{n_correct}/{len(results_df)}")
    st.caption("Precision: 63% | Recall: 75% | PR-AUC: 0.78")
    st.caption("(from held-out test set evaluation)")
    st.metric("Transactions Reviewed", len(st.session_state.reviewed))