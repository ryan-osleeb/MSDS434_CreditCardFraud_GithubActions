import pandas as pd

# Load the original dataset
df = pd.read_csv("creditcard.csv")

# Keep only fraud rows
fraud = df[df["Class"] == 1].copy()

# (optional) save to a new file
fraud.to_csv("creditcard_fraud.csv", index=False)

print(f"{len(fraud)} fraud rows saved to creditcard_fraud.csv")
