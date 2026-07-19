# Data Preprocessing - COVID-19 Dataset

โปรแกรมนี้เป็นส่วนหนึ่งของการเรียนรู้เรื่อง **Data Preprocessing** โดยใช้ข้อมูลผู้ติดเชื้อ COVID-19 ทั่วโลก เพื่อศึกษาการสำรวจข้อมูล การทำความสะอาดข้อมูล และการเตรียมข้อมูลก่อนนำไปใช้ใน Machine Learning

---

## Libraries ที่ใช้

### pandas
ใช้สำหรับโหลดและจัดการข้อมูลในรูปแบบตาราง (DataFrame)

```python
import pandas as pd
```

---

### matplotlib.pyplot
ใช้สร้างกราฟพื้นฐาน เช่น Histogram

```python
import matplotlib.pyplot as plt
```

---

### seaborn
ใช้สร้างกราฟที่สวยงามและแสดง Correlation Heatmap

```python
import seaborn as sns
```

---

### LabelEncoder
ใช้แปลงข้อมูลประเภทข้อความให้เป็นตัวเลข

```python
from sklearn.preprocessing import LabelEncoder
```

---

# Dataset

```python
df = pd.read_csv("time_series_covid19_confirmed_global.csv")
```

โหลดไฟล์ข้อมูล COVID-19 เข้าสู่ DataFrame

---

# LAB1 : Dataset Exploration

### Display Shape

```python
df.shape
```

แสดงจำนวนแถวและจำนวนคอลัมน์ของข้อมูล

---

### Display Data Types

```python
df.dtypes
```

แสดงชนิดข้อมูลของแต่ละคอลัมน์

---

### Display Summary Statistics

```python
df.describe()
```

แสดงสถิติพื้นฐาน เช่น

- Count
- Mean
- Std
- Min
- Max
- Quartiles

---

### Display Missing Values

```python
df.isnull().sum()
```

ตรวจสอบจำนวนข้อมูลที่หายไป (Missing Values)

---

### Display Duplicate Records

```python
df.duplicated().sum()
```

ตรวจสอบจำนวนข้อมูลซ้ำ

---

### Display Class Distribution

```python
df["Country/Region"].value_counts()
```

แสดงจำนวนข้อมูลของแต่ละประเทศ

---

# LAB2 : Data Visualization

### Histogram

```python
plt.hist(df.iloc[:, -1])
```

แสดงการกระจายตัวของจำนวนผู้ติดเชื้อในวันล่าสุด

---

### Correlation Heatmap

```python
sns.heatmap(numeric_df.corr())
```

แสดงความสัมพันธ์ของข้อมูลตัวเลขแต่ละคอลัมน์

---

### Mean

```python
df.iloc[:, -1].mean()
```

คำนวณค่าเฉลี่ยของจำนวนผู้ติดเชื้อ

---

### Median

```python
df.iloc[:, -1].median()
```

คำนวณค่ามัธยฐานของจำนวนผู้ติดเชื้อ

---

# Part 3 : Data Cleaning

### Missing Value Handling

```python
fillna()
```

แทนค่าที่หายไปด้วยข้อมูลที่กำหนด

---

### Duplicate Removal

```python
drop_duplicates()
```

ลบข้อมูลที่ซ้ำกันออก

---

### Incorrect Data Correction

```python
df.columns = df.columns.str.strip()
```

แก้ไขชื่อคอลัมน์ให้ถูกต้องโดยลบช่องว่าง

---

### Data Type Conversion

```python
astype(float)
```

แปลงชนิดข้อมูลให้เหมาะสม

---

# Part 4 : Feature Engineering

### Label Encoding

```python
LabelEncoder()
```

แปลงข้อมูลข้อความเป็นตัวเลข เช่น

Thailand → 150

Japan → 90

USA → 180

---

### One-Hot Encoding

```python
pd.get_dummies()
```

แปลงข้อมูลประเภทข้อความให้เป็นหลายคอลัมน์ที่มีค่า 0 และ 1

ตัวอย่าง

| Country | Thailand | Japan | USA |
|----------|-----------|--------|-----|
| Thailand | 1 | 0 | 0 |
| Japan | 0 | 1 | 0 |
| USA | 0 | 0 | 1 |

---

## วัตถุประสงค์

- ศึกษาการสำรวจข้อมูล (Data Exploration)
- วิเคราะห์ข้อมูลด้วยกราฟ (Data Visualization)
- ทำความสะอาดข้อมูล (Data Cleaning)
- เตรียมข้อมูลสำหรับ Machine Learning (Feature Engineering)