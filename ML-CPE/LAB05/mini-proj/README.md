## Dataset

Rice Image Dataset

อ้างอิง:

https://www.kaggle.com/code/pkdarabi/rice-classification-by-cnn/input

# SVM Image Recognition — Rice Varieties

โปรเจกต์นี้เป็นการจำแนกชนิดของข้าวจากรูปภาพโดยใช้ SVM (Support Vector Machine)

หลักการทำงานคืออ่านรูปภาพจาก Dataset แล้วนำมาแปลงเป็น Feature โดยปรับขนาดภาพให้เท่ากันและแปลงเป็นภาพ Grayscale จากนั้นแบ่งข้อมูลเป็น Train และ Test ใช้ PCA ในการลดจำนวน Feature และนำไป Train ด้วย SVM แบบ RBF Kernel

หลังจาก Train แล้วจะทำการวัด Accuracy, Classification Report และแสดง Confusion Matrix เพื่อดูผลการทำนายของโมเดล

## โครงสร้างโปรเจกต์

```text
SVM-Rice-Classification/
│
├── DATASET/
│   └── Rice_Image_Dataset/
│       ├── Arborio/
│       ├── Basmati/
│       ├── Jasmine/
│       └── ...
│
├── outputs/
│   ├── images.npy
│   ├── labels.npy
│   ├── classes.json
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   ├── confusion_matrix.png
│   └── prediction_sample.png
│
├── main.py
├── data_load.py
├── preprocess.py
├── split_data.py
├── svm_model.py
├── evaluate.py
├── test_svm.py
└── README.md
```

## ไฟล์หลัก

* `main.py`
  ไฟล์หลักสำหรับรันโปรเจกต์ ตั้งแต่โหลด Dataset, เตรียมข้อมูล, Train โมเดล ไปจนถึงประเมินผล

* `data_load.py`
  ใช้โหลดรูปภาพจากโฟลเดอร์ Dataset และสร้าง Label ของแต่ละ Class

* `preprocess.py`
  ใช้เตรียมรูปภาพ เช่น แปลงเป็น Grayscale, Resize และแปลงรูปภาพเป็น Feature

* `split_data.py`
  ใช้แบ่งข้อมูลออกเป็น Train และ Test โดยใช้ Stratified Split

* `svm_model.py`
  ใช้สร้างและ Train SVM รวมถึงฟังก์ชันสำหรับ Prediction

* `evaluate.py`
  ใช้คำนวณ Accuracy, Classification Report และสร้าง Confusion Matrix

* `test_svm.py`
  ใช้โหลดโมเดลที่ Train แล้ว และทดสอบ Prediction จากรูปภาพใน Test Set

## Dataset

Dataset ต้องแบ่งเป็นโฟลเดอร์ตามชนิดของข้าว เช่น

```text
DATASET/
└── Rice_Image_Dataset/
    ├── Arborio/
    ├── Basmati/
    ├── Jasmine/
    └── ...
```

ในแต่ละโฟลเดอร์จะเก็บรูปภาพของข้าวชนิดนั้น ๆ เช่น `.jpg`, `.png` เป็นต้น

ชื่อของโฟลเดอร์จะถูกนำมาใช้เป็น Class ของโมเดล

## การติดตั้ง

ใช้ Python 3.8 ขึ้นไป

ติดตั้ง Library ที่ใช้ด้วยคำสั่ง

```bash
pip install numpy opencv-python scikit-learn matplotlib joblib
```

โปรเจกต์นี้ทดสอบบน Python 3.13

## วิธีใช้งาน

### 1. เตรียม Dataset

นำ Dataset ไปไว้ที่

```text
DATASET/Rice_Image_Dataset
```

ถ้าใช้ Path อื่น สามารถแก้ Path ได้ใน `main.py`

### 2. ตั้งค่าพารามิเตอร์

ตัวอย่างค่าที่สามารถปรับได้ใน `main.py`

```python
IMG_SIZE = 100
TEST_SIZE = 0.2
MAX_PER_CLASS = None
```

รายละเอียดของแต่ละค่า

| Parameter       | รายละเอียด                       | ค่าเริ่มต้น |
| --------------- | -------------------------------- | ----------- |
| `IMG_SIZE`      | ขนาดภาพหลัง Resize               | `100`       |
| `TEST_SIZE`     | สัดส่วนข้อมูลที่ใช้เป็น Test Set | `0.2`       |
| `MAX_PER_CLASS` | จำนวนภาพสูงสุดที่ใช้ต่อ Class    | `None`      |

ถ้า `MAX_PER_CLASS = None` จะใช้รูปภาพทั้งหมดที่มีในแต่ละ Class

### 3. Train Model

รันคำสั่ง

```bash
python main.py
```

โปรแกรมจะทำงานตามลำดับ

```text
Load Dataset
    ↓
Preprocess
    ↓
Train/Test Split
    ↓
Scaling
    ↓
PCA
    ↓
SVM Training
    ↓
Evaluation
```

### 4. ทดสอบ Prediction

หลังจาก Train โมเดลเสร็จแล้ว สามารถรัน

```bash
python test_svm.py
```

เพื่อดูตัวอย่างผลการทำนาย

ผลลัพธ์จะถูกบันทึกไว้ที่

```text
outputs/prediction_sample.png
```

## SVM และ PCA

ในโปรเจกต์นี้ใช้ SVM แบบ RBF Kernel โดยก่อนนำข้อมูลเข้า SVM จะมีการ Scale และลด Dimension ด้วย PCA

การทำงานโดยรวมเป็น

```text
Image
 ↓
Grayscale
 ↓
Resize
 ↓
Flatten
 ↓
Scaling
 ↓
PCA
 ↓
SVM (RBF)
```

ตัวอย่างเช่น ถ้ากำหนดขนาดภาพเป็น `100 x 100` จะมี Pixel ทั้งหมด 10,000 ค่า ซึ่งเมื่อนำไปใช้เป็น Feature โดยตรงจะมี Dimension ค่อนข้างสูง

จึงใช้ PCA เพื่อลดจำนวน Feature ก่อนนำไป Train SVM

## ผลลัพธ์

หลังจากรัน `main.py` จะมีไฟล์ถูกสร้างขึ้นในโฟลเดอร์ `outputs`

```text
outputs/
├── images.npy
├── labels.npy
├── classes.json
├── X_train.npy
├── X_test.npy
├── y_train.npy
├── y_test.npy
├── svm_model.pkl
├── scaler.pkl
├── confusion_matrix.png
└── prediction_sample.png
```

ไฟล์ที่สำคัญ เช่น

* `svm_model.pkl` — โมเดล SVM ที่ Train แล้ว
* `scaler.pkl` — Scaler ที่ใช้กับข้อมูล
* `classes.json` — รายชื่อ Class
* `confusion_matrix.png` — ผล Confusion Matrix
* `prediction_sample.png` — ตัวอย่างผล Prediction

## ข้อควรระวัง

ถ้า Dataset มีรูปจำนวนมาก การโหลดรูปทั้งหมดเข้าหน่วยความจำอาจใช้ RAM ค่อนข้างเยอะ

ถ้าเจอปัญหา RAM ไม่พอ สามารถกำหนดจำนวนรูปสูงสุดต่อ Class ได้ เช่น

```python
MAX_PER_CLASS = 1000
```

หรือสามารถปรับโค้ดให้โหลดข้อมูลเป็น Batch ในภายหลังได้

## สิ่งที่สามารถพัฒนาต่อได้

* เพิ่ม argparse เพื่อกำหนด Parameter ผ่าน Command Line
* ทดลองปรับค่า `C` และ `gamma` ของ SVM
* เปรียบเทียบ Kernel แบบ Linear กับ RBF
* เพิ่ม Data Augmentation
* เพิ่มการบันทึกผลการทดลอง
* ทำหน้าเว็บสำหรับ Upload รูปภาพแล้วทำนายชนิดข้าว
* ทดลองเปรียบเทียบผลกับ CNN หรือ Transfer Learning


