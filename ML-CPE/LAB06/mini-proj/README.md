# คู่มือโค้ด `mini-proj`

โฟลเดอร์นี้มี pipeline สำหรับจำแนกภาพเมล็ดกาแฟเป็น `bad bean` และ `good bean`
ด้วย TensorFlow/Keras โดยอ่าน dataset จากโฟลเดอร์ `../dataset` ซึ่งเป็น dataset
จาก [Coffee defect - v2 บน Roboflow](https://universe.roboflow.com/tirta-inovan-lgpth/coffee-defect/dataset/2)

## วิธีใช้งาน

ติดตั้ง dependency จากนั้นรัน:

```bash
pip install numpy opencv-python matplotlib scikit-learn tensorflow
python main.py
python test_nn.py
```

ควรรันคำสั่งจากโฟลเดอร์นี้ โดย `main.py` จะสร้างหรือเขียนผลลัพธ์ใน `outputs/`.

## หน้าที่ของแต่ละไฟล์

- `main.py` ควบคุมลำดับตั้งแต่อ่านข้อมูล preprocess ฝึก ทดสอบ และประเมินผล
- `data_loader.py` อ่านภาพและ label จาก `_classes.csv` ใน `train/`, `valid/`, `test/`
- `preprocessing.py` แปลง BGR เป็น RGB และ resize ภาพเป็น `100x100`
- `nn_model.py` สร้างและฝึก MLP พร้อมบันทึก `nn_model.keras` และ `history.json`
- `evaluate.py` แสดง accuracy, classification report, confusion matrix และกราฟการฝึก
- `test_nn.py` โหลดโมเดลและสุ่มภาพจาก test set เพื่อแสดงผล prediction
- `split_data.py` มีฟังก์ชันแบ่ง dataset แบบ stratified สำหรับกรณีที่ต้องแบ่งข้อมูลใหม่

รายละเอียด dataset, ผลลัพธ์ และคำอธิบาย pipeline ฉบับเต็มอยู่ที่ [README หลัก](../README.md)
## 
