

# การจำแนกประเภทขยะจากภาพ

โปรเจกต์นี้ใช้โมเดล CNN ตระกูล VGG จำแนกภาพขยะ 6 ประเภท และเปรียบเทียบผลของโมเดล 4 ขนาด ได้แก่ VGG16, VGG Small, VGG Medium และ VGG Tiny

## ชุดข้อมูล

ใช้ชุดข้อมูล [Garbage classification dataset บน Kaggle](https://www.kaggle.com/datasets/manonstr/tipe-webscraping) โดยดาวน์โหลดและจัดไฟล์ภาพให้อยู่ในโครงสร้างต่อไปนี้ก่อนเริ่มรัน:

```text
dataset/
├── entrainement/
│   ├── Bouteille_plastique/
│   ├── Brique_en_carton/
│   ├── Emballage_metallique/
│   ├── Ordure_menagere/
│   ├── Papier_Carton/
│   └── Verre/
└── test/
    ├── Bouteille_plastique/
    ├── Brique_en_carton/
    ├── Emballage_metallique/
    ├── Ordure_menagere/
    ├── Papier_Carton/
    └── Verre/
```

ชื่อโฟลเดอร์ใน `entrainement` และ `test` ต้องตรงกัน ภาพจะถูกปรับขนาดเป็น 50 × 50 พิกเซล ส่วนข้อมูลใน `entrainement` จะถูกแบ่งเป็นชุดฝึกและชุด validation ในสัดส่วน 80:20 ข้อมูลใน `test` ใช้สำหรับประเมินผลหลังฝึก

## ติดตั้งและเริ่มใช้งาน

ต้องมี Python และติดตั้งไลบรารีที่โปรเจกต์ใช้:

```bash
pip install numpy opencv-python scikit-learn matplotlib tensorflow
```

จากโฟลเดอร์โปรเจกต์ รันคำสั่ง:

```bash
python main.py
```

สคริปต์จะฝึกโมเดลทั้ง 4 แบบ โดยตั้งค่าเริ่มต้นไว้ที่ไม่เกิน 20 epochs และ batch size 64 การฝึกอาจใช้เวลานาน ขึ้นอยู่กับจำนวนภาพและอุปกรณ์ที่ใช้

## ผลลัพธ์

ไฟล์ที่ได้จะถูกบันทึกใน `outputs/` เช่น โมเดลและประวัติการฝึกในโฟลเดอร์ของแต่ละสถาปัตยกรรม, `metrics.json` สำหรับ accuracy บนชุด test และกราฟ confusion matrix, training history และการเปรียบเทียบโมเดล

## การปรับค่า

ค่าหลักสำหรับปรับการทดลองอยู่ตอนต้นของ `main.py` ได้แก่ `IMG_SIZE`, `VAL_SIZE`, `EPOCHS`, `BATCH_SIZE` และ `MAX_PER_CLASS` หากต้องการใช้ภาพทั้งหมด ให้คง `MAX_PER_CLASS = None` ไว้
