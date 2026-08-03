# LAB 04 - Wine Dataset: Classification และ Clustering

โปรเจกต์นี้เป็นการทดลอง Machine Learning บนชุดข้อมูล Wine Dataset เพื่อแสดงให้เห็นถึง 2 แบบการเรียนรู้หลักคือ

1. Classification ด้วยเทคนิค K-Nearest Neighbors (KNN)
2. Clustering ด้วยเทคนิค K-Means

Dataset ที่ใช้มาจาก Kaggle: https://www.kaggle.com/datasets/tawfikelmetwally/wine-dataset

---

## 1. โครงสร้างโปรเจกต์

- classification/: โค้ดสำหรับงาน Classification
  - main.py: ไฟล์รันหลักสำหรับเทส KNN
  - data_loader.py: โหลดและเตรียมข้อมูล
  - knn_model.py: สร้างโมเดล KNN
  - evaluate.py: ประเมินผลและบันทึกผลลัพธ์

- clustering/: โค้ดสำหรับงาน Clustering
  - main.py: ไฟล์รันหลักสำหรับ K-Means
  - data_loader.py: โหลดและปรับมาตราส่วนข้อมูล
  - kmeans_tf.py: สร้างและฝึกโมเดล K-Means
  - knn_tools.py: คำนวณค่า WCSS สำหรับ Elbow Method
  - visualize.py: สร้างกราฟและบันทึกไฟล์ CSV

- data-wine/: ไฟล์ข้อมูลต้นฉบับ
  - Wine dataset.csv

---

## 2. Dataset ที่ใช้

ไฟล์ข้อมูลอยู่ที่:
- data-wine/Wine dataset.csv

ข้อมูลนี้มีลักษณะดังนี้
- คอลัมน์แรกเป็น Label หรือ Class ของไวน์
- คอลัมน์ที่เหลือเป็น Feature ของข้อมูล เช่น ปริมาณสารเคมีต่าง ๆ ที่ใช้ในการจำแนกประเภทของไวน์

ในการทำงานจริง โค้ดจะใช้คอลัมน์แรกเป็น y (target) และคอลัมน์ที่เหลือเป็น X (input features)

---

## 3. ส่วนที่ 1: Classification ด้วย KNN

### วัตถุประสงค์
การจำแนกประเภทของข้อมูลด้วยวิธี K-Nearest Neighbors โดยใช้ค่าของ k ที่แตกต่างกันเพื่อดูว่า k ไหนให้ผลลัพธ์ดีที่สุด

### กระบวนการทำงาน

1. โหลดข้อมูลจากไฟล์ CSV
2. แยกข้อมูลเป็น train/test โดยใช้ train_test_split
3. ปรับมาตราส่วนข้อมูลด้วย StandardScaler เพื่อให้คุณสมบัติของข้อมูลอยู่ในช่วงเดียวกัน
4. สร้างโมเดล KNN สำหรับค่า k ต่าง ๆ คือ 3, 5, 7
5. ประเมินความแม่นยำด้วย accuracy_score
6. สร้าง Confusion Matrix และบันทึกผลลัพธ์ลงไฟล์

### ไฟล์ที่เกี่ยวข้อง
- classification/data_loader.py
  - ทำหน้าที่อ่านข้อมูลและเตรียมตัวแปร X, y สำหรับ train/test
- classification/knn_model.py
  - สร้างโมเดล KNeighborsClassifier
- classification/evaluate.py
  - คำนวณความแม่นยำ ตรวจสอบผลการทำนาย และบันทึกผลเป็น CSV
- classification/main.py
  - รันโมเดลหลายค่า k และเลือกค่า k ที่ดีที่สุด

### ผลลัพธ์ที่ได้
เมื่อรันคำสั่งด้านล่าง จะสร้างไฟล์ผลลัพธ์ดังนี้
- classification/outputs/01_k_curve.png: กราฟแสดงความสัมพันธ์ระหว่าง k กับ Accuracy
- classification/outputs/02_confusion_matrix.png: Confusion Matrix ของผลทดสอบ
- classification/outputs/predictions.csv: ตารางที่เก็บผล Actual และ Prediction

### วิธีรัน
```bash
python classification/main.py
```

---

## 4. ส่วนที่ 2: Clustering ด้วย K-Means

### วัตถุประสงค์
การแบ่งข้อมูลออกเป็นกลุ่มต่าง ๆ โดยไม่ใช้ป้ายกำกับ (label) จากข้อมูลเดิม เพื่อดูว่าข้อมูลมีโครงสร้างกลุ่มอย่างไร

### กระบวนการทำงาน

1. โหลดข้อมูลและปรับมาตราส่วนข้อมูลด้วย StandardScaler
2. ใช้ Elbow Method เพื่อดูว่า k ที่เหมาะสมควรเป็นเท่าใด
3. คำนวณ WCSS (Within-Cluster Sum of Squares) สำหรับจำนวนกลุ่มตั้งแต่ 1 ถึง 10
4. สร้างกราฟ Elbow เพื่อช่วยเลือกจำนวนกลุ่มที่เหมาะสม
5. ใช้ K-Means กับจำนวนกลุ่ม 3 กลุ่ม
6. บันทึกผลการจัดกลุ่มและสร้างกราฟการกระจายของกลุ่ม

### ไฟล์ที่เกี่ยวข้อง
- clustering/data_loader.py
  - อ่านข้อมูลและเตรียมข้อมูลสำหรับ clustering
- clustering/kmeans_tf.py
  - สร้างและฝึกโมเดล K-Means
- clustering/knn_tools.py
  - คำนวณ WCSS สำหรับ Elbow Method
- clustering/visualize.py
  - สร้างภาพกราฟและบันทึกข้อมูลผล clustering
- clustering/main.py
  - รัน pipeline ของ clustering ทั้งหมด

### ผลลัพธ์ที่ได้
เมื่อรันคำสั่งด้านล่าง จะสร้างไฟล์ผลลัพธ์ดังนี้
- clustering/outputs/01_elbow.png: กราฟ Elbow Method
- clustering/outputs/02_clusters.png: กราฟแสดงผลลัพธ์การจัดกลุ่ม
- clustering/outputs/clustered_wine.csv: ข้อมูลที่มีคอลัมน์ Cluster เพิ่มเติม
- clustering/outputs/cluster_summary.csv: สรุปจำนวนข้อมูลในแต่ละ Cluster

### วิธีรัน
```bash
python clustering/main.py
```

---

## 5. คำอธิบายสั้น ๆ ของเทคนิคที่ใช้

### KNN (K-Nearest Neighbors)
KNN เป็นอัลกอริทึม Classification แบบไม่ต้องฝึกแบบซับซ้อนมากนัก เพราะจะพิจารณาเพียงข้อมูลที่ใกล้เคียงที่สุดกับข้อมูลใหม่ แล้วเลือกคลาสที่มีจำนวนเพื่อนบ้านมากที่สุด

### K-Means
K-Means เป็นอัลกอริทึม Clustering ที่แบ่งข้อมูลเป็นกลุ่มตามระยะห่างจาก centroid ของแต่ละกลุ่ม โดยจะวนซ้ำจนกลุ่มไม่เปลี่ยนแปลงมากนัก

---

## 6. สิ่งที่ต้องติดตั้ง

ติดตั้ง dependency ด้วยคำสั่ง:

```bash
pip install -r requirements.txt
```

Dependencies ที่ใช้:
- pandas
- numpy
- scikit-learn
- matplotlib

---

## 7. สรุป

โปรเจกต์นี้แสดงให้เห็นว่า Machine Learning สามารถแบ่งออกเป็น 2 แนวทางหลักคือ
- Classification: ใช้ข้อมูลที่มี label เพื่อเรียนรู้การทำนายผลลัพธ์
- Clustering: ใช้ข้อมูลที่ไม่มี label เพื่อค้นหากลุ่มที่ซ่อนอยู่

ทั้งสองส่วนถูกนำมาใช้งานร่วมกันในชุดข้อมูลเดียวกันและช่วยให้เราเห็นภาพของกระบวนการทำงานของ ML แบบพื้นฐานอย่างชัดเจน

