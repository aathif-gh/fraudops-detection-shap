import pandas as pd

df = pd.read_csv("creditcard.csv")
df = df.sort_values("Time").reset_index(drop=True)
n = len(df)
val_end = int(n * 0.85)
test_df = df.iloc[val_end:]

test_df.to_csv("creditcard_test.csv", index=False)
print(f"Saved {len(test_df)} test-set rows")