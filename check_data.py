import pandas as pd

df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

print("Number of records:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nChurn count:")
print(df["Churn"].value_counts())