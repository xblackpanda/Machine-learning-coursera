import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# โหลด Dataset
df = pd.read_csv("time_series_covid19_confirmed_global.csv")

# เลือกเฉพาะคอลัมน์ตัวเลข
numeric_df = df.select_dtypes(include='number')

# 1. Histogram (จำนวนผู้ติดเชื้อวันล่าสุด)
plt.figure(figsize=(8,5))
plt.hist(df.iloc[:, -1], bins=20)
plt.title("Histogram of Confirmed Cases")
plt.xlabel("Confirmed Cases")
plt.ylabel("Frequency")
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3. Mean
print("Mean =", df.iloc[:, -1].mean())

# 4. Median
print("Median =", df.iloc[:, -1].median())



# 1. Missing Value Handling
print("Missing ก่อนแก้")
print(df.isnull().sum())

df["Province/State"] = df["Province/State"].fillna("Unknown")

print("\nMissing หลังแก้")
print(df.isnull().sum())

# 2. Duplicate Removal
print("\nDuplicate ก่อนลบ =", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate หลังลบ =", df.duplicated().sum())

# 3. Incorrect Data Correction
# ลบช่องว่างหน้าหลังชื่อคอลัมน์
df.columns = df.columns.str.strip()

# 4. Data Type Conversion
df["Lat"] = df["Lat"].astype(float)
df["Long"] = df["Long"].astype(float)

print("\nData Types")
print(df.dtypes)


from sklearn.preprocessing import LabelEncoder

# 1. Label Encoding
encoder = LabelEncoder()
df["Country_Label"] = encoder.fit_transform(df["Country/Region"])

print(df[["Country/Region","Country_Label"]].head())

# 2. One-Hot Encoding
onehot = pd.get_dummies(df["Country/Region"], prefix="Country")

df = pd.concat([df, onehot], axis=1)

print(df.head())