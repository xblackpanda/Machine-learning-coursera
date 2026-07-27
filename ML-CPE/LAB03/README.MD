# Machine Learning Labs

โครงการนี้เป็นชุดงานฝึก Machine Learning ที่ใช้ข้อมูลทั้งแบบตารางและภาพ เพื่อฝึกพื้นฐานการสร้างโมเดลและดูผลลัพธ์แบบง่าย ๆ

## แหล่งข้อมูลที่ใช้

- Face-Recognition dataset: จาก GitHub repo [EvilPort2/Face-Recognition]
(https://github.com/EvilPort2/Face-Recognition/blob/master/dataset.csv)
  - ใช้เป็นข้อมูลตัวอย่างสำหรับทดลองทำนายและจำแนกข้อมูลแบบพื้นฐาน

- UTKFace dataset: จาก Kaggle [UTKFace]
(https://www.kaggle.com/datasets/jangedoo/utkface-new)
  - ใช้สำหรับฝึกโมเดลจากรูปภาพ โดยเอาอายุและเพศจากชื่อไฟล์มาเป็น label

## คำอธิบายแต่ละไฟล์ / Lab

### 1. lab1.py
เป็น Lab การทำนายค่าแบบ Regression
- ใช้โมเดล Simple Linear Regression และ Multiple Linear Regression
- ฝึกให้ดูว่าเมื่อมีตัวแปรหลายตัว จะทำนายค่าอายุได้ดีแค่ไหน
- ผลลัพธ์จะดูจากค่า R² และค่า error ของโมเดล

### 2. lab2.py
เป็น Lab การจำแนกข้อมูลแบบ Classification
- ใช้ Logistic Regression เพื่อจำแนกข้อมูลเป็นกลุ่มต่าง ๆ
- เหมาะสำหรับฝึกเรื่อง confusion matrix, accuracy และการดูผลลัพธ์ของการคาดเดา

### 3. lab3.py
เป็น Lab สำหรับเปรียบเทียบโมเดลต่าง ๆ
- จะเก็บผลลัพธ์จากโมเดลที่ใช้ในแต่ละงานมาเปรียบเทียบกัน
- เหมาะกับการดูว่าโมเดลไหนให้ผลลัพธ์ดีกว่ากันในสถานการณ์ต่าง ๆ


### 4. image_training.py
เป็นไฟล์สำหรับฝึกโมเดลจากภาพจริง
- อ่านไฟล์รูปภาพจากโฟลเดอร์ UTKFace
- แยกข้อมูลอายุและเพศจากชื่อไฟล์
- สร้างโมเดลเพื่อทำนายอายุและจำแนกเพศจากภาพ

## สรุปสั้น ๆ
โครงการนี้เหมาะสำหรับคนเริ่มเรียน Machine Learning เพราะจะได้เห็นการทำงานของ regression, classification และการใช้ภาพเป็นข้อมูลจริงแบบง่าย ๆ โดยไม่ต้องซับซ้อนเกินไป

