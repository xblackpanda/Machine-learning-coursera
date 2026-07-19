import pandas as pd

df = pd.read_csv("time_series_covid19_confirmed_global.csv")


# 1. Load Dataset

print("โหลดข้อมูลสำเร็จ\n")

# 2. Display Shape

print("Shape")
print(df.shape)

# 3. Display Data Types

print("\nData Types")
print(df.dtypes)


# 4. Display Summary Statistics

print("\nSummary Statistics")
print(df.describe())

# 5. Display Missing Values

print("\nMissing Values")
print(df.isnull().sum())

# 6. Display Duplicate Records

print("\nDuplicate Records")
print("จำนวนข้อมูลซ้ำ =", df.duplicated().sum())

# 7. Display Class Distribution

print("\nClass Distribution (Country/Region)")
print(df["Country/Region"].value_counts())