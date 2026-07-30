from fastapi import FastAPI
from pydantic import BaseModel, create_model
import joblib
import shap
import pandas as pd

app = FastAPI(title="Fraud Detection API")

# Load everything we saved from training
model = joblib.load("fraud_model.joblib")
threshold = joblib.load("threshold.joblib")
feature_cols = joblib.load("feature_cols.joblib")
explainer = shap.TreeExplainer(model)

# Dynamically build a request schema with one field per feature (V1-V28, Time, Amount)
fields = {col: (float, ...) for col in feature_cols}
Transaction = create_model("Transaction", **fields)

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert incoming request into the same row format the model expects
    row = pd.DataFrame([transaction.dict()])[feature_cols]

    prob = model.predict_proba(row)[:, 1][0]
    is_fraud = bool(prob >= threshold)

    # SHAP explanation for this specific transaction
    shap_values = explainer.shap_values(row)[0]
    top_features = sorted(
        zip(feature_cols, shap_values), key=lambda x: abs(x[1]), reverse=True
    )[:5]

    return {
        "fraud_probability": round(float(prob), 4),
        "is_fraud": is_fraud,
        "threshold_used": round(float(threshold), 4),
        "top_contributing_features": [
            {"feature": f, "impact": round(float(v), 4)} for f, v in top_features
        ]
    }

@app.get("/")
def root():
    return {"status": "Fraud detection API is running"}