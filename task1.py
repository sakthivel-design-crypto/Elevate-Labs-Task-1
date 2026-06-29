import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())
# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column
df.drop("Cabin", axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

print("\nAfter Encoding:")
print(df.head())

scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

print("\nAfter Scaling:")
print(df.head())


plt.figure(figsize=(6,4))
sns.boxplot(x=df["Fare"])
plt.title("Fare Boxplot")
plt.show()
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)]

print("\nShape After Removing Outliers:")
print(df.shape)
df.to_csv("Titanic-Cleaned.csv", index=False)

print("\nDataset cleaned successfully!")
print("Saved as Titanic-Cleaned.csv")