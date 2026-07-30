import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, classification_report
from sklearn.metrics import precision_recall_curve
import shap
import joblib

# Load the dataset
df = pd.read_csv("creditcard.csv")

print(df.shape)
print(df["Class"].value_counts())

# Sorting by time (should already be sorted, but this guarantees it)
df = df.sort_values("Time").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print(f"Train: {len(train_df)} rows, {train_df['Class'].sum()} fraud")
print(f"Val:   {len(val_df)} rows, {val_df['Class'].sum()} fraud")
print(f"Test:  {len(test_df)} rows, {test_df['Class'].sum()} fraud")





feature_cols = [c for c in df.columns if c not in ["Class"]]

X_train, y_train = train_df[feature_cols], train_df["Class"]
X_val, y_val = val_df[feature_cols], val_df["Class"]
X_test, y_test = test_df[feature_cols], test_df["Class"]

# SMOTE creates synthetic fraud examples by interpolating between real
# fraud cases' feature values — NOT just duplicating them.
# sampling_strategy=0.1 means: bring fraud up to 10% of the training set,
# not a full 50/50 — going all the way to balanced tends to overfit on
# synthetic patterns that don't reflect how rare fraud really is.
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.sum()} fraud / {len(y_train)} total")
print(f"After SMOTE:  {y_train_res.sum()} fraud / {len(y_train_res)} total")

model = XGBClassifier(
    max_depth=5,
    n_estimators=200,
    learning_rate=0.1,
    eval_metric="aucpr",
    random_state=42
)

model.fit(X_train_res, y_train_res)

# Evaluate on validation set — note: we NEVER apply SMOTE to val/test,
# only to training data, since val/test must reflect real-world distribution
val_probs = model.predict_proba(X_val)[:, 1]
val_pr_auc = average_precision_score(y_val, val_probs)
print(f"\nValidation PR-AUC: {val_pr_auc:.4f}")

val_preds = (val_probs >= 0.5).astype(int)
print(classification_report(y_val, val_preds))




# Assume: a missed fraud costs $500 on average, a false alarm costs $20
# (manual review time / customer friction). Adjust these if you want —
# the ratio matters more than the exact numbers.
COST_FN = 500
COST_FP = 20

precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

best_thresh, best_cost = 0.5, float("inf")
for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
    tp = r * y_val.sum()
    fn = y_val.sum() - tp
    fp = tp * (1 - p) / p if p > 0 else 0
    cost = fn * COST_FN + fp * COST_FP
    if cost < best_cost:
        best_cost, best_thresh = cost, t

print(f"\nBest threshold: {best_thresh:.4f} (estimated cost: ${best_cost:.0f})")

# Re-evaluate with this threshold instead of 0.5
val_preds_optimal = (val_probs >= best_thresh).astype(int)
print(classification_report(y_val, val_preds_optimal))



explainer = shap.TreeExplainer(model)

# Explain the first 3 transactions in the validation set that were flagged as fraud
flagged_idx = val_df[val_preds_optimal == 1].index[:3]
X_flagged = X_val.loc[flagged_idx]

shap_values = explainer.shap_values(X_flagged)

for i, idx in enumerate(flagged_idx):
    print(f"\n--- Transaction {idx} (flagged as fraud) ---")
    feature_impact = sorted(
        zip(feature_cols, shap_values[i]),
        key=lambda x: abs(x[1]), reverse=True
    )
    for feat, val in feature_impact[:5]:
        direction = "pushed toward FRAUD" if val > 0 else "pushed toward LEGIT"
        print(f"  {feat}: {val:+.3f}  ({direction})")


# Final evaluation on the TEST set 

test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_thresh).astype(int)

test_pr_auc = average_precision_score(y_test, test_probs)
print(f"\n=== FINAL TEST SET RESULTS ===")
print(f"Test PR-AUC: {test_pr_auc:.4f}")
print(classification_report(y_test, test_preds))

# Save everything the API/dashboard will need
joblib.dump(model, "fraud_model.joblib")
joblib.dump(best_thresh, "threshold.joblib")
joblib.dump(feature_cols, "feature_cols.joblib")
print("\nSaved: fraud_model.joblib, threshold.joblib, feature_cols.joblib")