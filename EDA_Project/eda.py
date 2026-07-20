import pandas as pd

# Load dataset
df = pd.read_csv("data/iris.csv")

print("========== First Five Records ==========")
print(df.head())

print("\n========== Dataset Information ==========")
df.info()

print("\n========== Summary Statistics ==========")
print(df.describe())

print("\n========== Missing Values ==========")
print(df.isnull().sum())

print("\n========== Correlation Matrix ==========")
print(df.corr(numeric_only=True))