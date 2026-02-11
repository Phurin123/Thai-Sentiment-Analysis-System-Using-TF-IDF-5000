# Thai Sentiment Analysis System Using TF-IDF

ระบบวิเคราะห์ความรู้สึกภาษาไทยแบบ Multi-Model โดยใช้ **TF-IDF** พร้อม Web UI สำหรับเปรียบเทียบประสิทธิภาพของโมเดลต่างๆ แบบ A/B Testing

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 สารบัญ

- [ภาพรวมโปรเจค](#ภาพรวมโปรเจค)
- [ฟีเจอร์หลัก](#ฟีเจอร์หลัก)
- [เทคโนโลยีที่ใช้](#เทคโนโลยีที่ใช้)
- [โครงสร้างโปรเจค](#โครงสร้างโปรเจค)
- [Quick Start](#quick-start)
- [การติดตั้ง](#การติดตั้ง)
- [วิธีรันโปรเจค](#วิธีรันโปรเจค)
- [การเทรนโมเดล](#การเทรนโมเดล)
- [API Documentation](#api-documentation)
- [วิธี Deploy บน Render](#วิธี-deploy-บน-render)
- [โมเดลที่รองรับ](#โมเดลที่รองรับ)
- [Troubleshooting](#troubleshooting)

---

## 🎯 ภาพรวมโปรเจค

ระบบนี้พัฒนาขึ้นเพื่อวิเคราะห์ความรู้สึก (Sentiment Analysis) ของข้อความภาษาไทย โดยจำแนกออกเป็น 3 ประเภท:
- **POSITIVE** (บวก) 😊
- **NEGATIVE** (ลบ) 😠
- **NEUTRAL** (กลาง) 😐

ระบบรองรับการเปรียบเทียบผลลัพธ์จาก **6 โมเดล Machine Learning** พร้อมกัน (A/B Testing) และมีระบบ Feedback เพื่อปรับปรุงความแม่นยำ

---

## ✨ ฟีเจอร์หลัก

✅ **Multi-Model Support**: รองรับ 6 โมเดล TF-IDF (Logistic Regression, Linear SVM, Random Forest, Naive Bayes, LightGBM, Extra Trees)  
✅ **A/B Testing UI**: เปรียบเทียบประสิทธิภาพของโมเดลต่างๆ ในหน้าเดียว  
✅ **Explainable AI**: แสดงคำสำคัญที่มีอิทธิพลต่อการทำนาย (Important Words)  
✅ **Feedback System**: รวบรวม feedback จากผู้ใช้เพื่อปรับปรุงโมเดล  
✅ **Error Tracking**: ติดตามและแสดงข้อผิดพลาดที่เกิดขึ้นจากการทำนาย  
✅ **RESTful API**: API endpoints สำหรับการ integrate กับระบบอื่น  
✅ **Real-time Analysis**: วิเคราะห์ความรู้สึกแบบ real-time พร้อมแสดงค่า latency

---

## 🛠️ เทคโนโลยีที่ใช้

### Backend & ML
- **FastAPI** - Modern web framework สำหรับสร้าง API
- **Uvicorn** - ASGI server สำหรับรัน FastAPI
- **scikit-learn** - Machine learning library สำหรับโมเดล TF-IDF
- **LightGBM** - Gradient boosting framework จาก Microsoft
- **pythainlp** - Thai NLP library

### Frontend
- **Bootstrap 5** - CSS framework
- **Vanilla JavaScript** - ไม่ใช้ framework เพิ่มเติม

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **joblib** - Model serialization

---

## 📁 โครงสร้างโปรเจค

```
Thai-Sentiment-Analysis-System-Using-TF-IDF/
│
├── app.py                          # 🚀 FastAPI main application
├── requirements.txt                # 📦 Python dependencies
├── information.txt                 # ℹ️ Quick start info
│
├── data/                           # 📊 Training datasets
│   ├── 1.synthetic_wisesight_like_thai_sentiment_5000.csv
│   ├── 1.synthetic_wisesight_like_thai_sentiment_100k.csv
│   └── error_examples*.csv         # Misclassified examples
│
├── models_regress/                 # 🤖 Logistic Regression models
├── models_linear/                  # 🤖 Linear SVM models
├── models_tree/                    # 🌳 Random Forest models
├── models_nb/                      # 🤖 Naive Bayes models
├── models_lgbm/                    # 💡 LightGBM models
├── models_et/                      # 🌲 Extra Trees models
│
├── templates/                      # 🎨 HTML templates
│   ├── index.html                  # Main UI page
│   └── errors.html                 # Error tracking page
│
├── static/                         # 🎨 Static files
│   └── style.css
│
└── Training Scripts:
    ├── Regress_train.py            # เทรน Logistic Regression
    ├── Renear_train.py             # เทรน Linear SVM
    ├── Random Forest_train.py      # เทรน Random Forest
    ├── naivebay.py                 # เทรน Naive Bayes
    ├── lightbgm.py                 # เทรน LightGBM
    └── extratree.py                # เทรน Extra Trees
```

---

## 🚀 Quick Start

**ติดตั้งและรันโปรเจคอย่างรวดเร็ว:**

```bash
# 1. Clone โปรเจค
git clone https://github.com/Phurin123/Thai-Sentiment-Analysis-System-Using-TF-IDF.git
cd Thai-Sentiment-Analysis-System-Using-TF-IDF

# 2. สร้าง Virtual Environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# หรือ source venv/bin/activate  # macOS/Linux

# 3. ติดตั้ง Dependencies
pip install -r requirements.txt

# 4. รัน Development Server
uvicorn app:app --reload

# 5. เปิดเบราว์เซอร์ไปที่
# http://127.0.0.1:8000/
```

---

## 📥 การติดตั้ง

### ความต้องการของระบบ

- **Python** 3.8 หรือสูงกว่า
- **pip** (Python package manager)
- **Virtual Environment** (แนะนำ)
- **RAM**: อย่างน้อย 4GB
- **Disk Space**: อย่างน้อย 1GB

### ขั้นตอนการติดตั้ง

#### 1. Clone โปรเจค

```bash
git clone https://github.com/Phurin123/Thai-Sentiment-Analysis-System-Using-TF-IDF.git
cd Thai-Sentiment-Analysis-System-Using-TF-IDF
```

#### 2. สร้าง Virtual Environment (แนะนำ)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **หมายเหตุ**: การติดตั้งอาจใช้เวลา 2-5 นาที ขึ้นอยู่กับความเร็วอินเทอร์เน็ต

#### 4. เตรียมข้อมูลและโมเดล

ตรวจสอบว่ามีโฟลเดอร์โมเดลและไฟล์ที่จำเป็น:

```
models_regress/
  ├── vectorizer_*.joblib
  └── sentiment_model_*.joblib

models_linear/
models_tree/
models_nb/
models_lgbm/
models_et/
```

> 💡 **คำแนะนำ**: ถ้ายังไม่มีโมเดล ให้รันสคริปต์เทรนโมเดลก่อน (ดูในส่วน [การเทรนโมเดล](#การเทรนโมเดล))

---

## 🏃 วิธีรันโปรเจค

### รัน Development Server

```bash
uvicorn app:app --reload
```

**หรือ**

```bash
python -m uvicorn app:app --reload
```

### เข้าถึงเว็บแอป

เปิดเบราว์เซอร์และไปที่:

```
http://127.0.0.1:8000/
```

### ตัวเลือกการรันเพิ่มเติม

#### กำหนด Port และ Host

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

#### รันโหมด Production (ไม่มี --reload)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### ตรวจสอบสถานะระบบ

เข้าไปที่:
```
http://127.0.0.1:8000/health
```

จะได้ response:
```json
{
  "status": "ok",
  "baseline_a": true,
  "available_models": ["linear", "rf", "nb", "lgbm", "et"]
}
```

---

## 🎓 การเทรนโมเดล

### ข้อมูลสำหรับการเทรน

โปรเจคนี้ใช้ dataset จากโฟลเดอร์ `data/`:
- `1.synthetic_wisesight_like_thai_sentiment_5000.csv` (5,000 รายการ)
- `1.synthetic_wisesight_like_thai_sentiment_100k.csv` (100,000 รายการ)

รูปแบบข้อมูล:
```csv
text,sentiment
"สินค้าดีมาก ส่งไว","POSITIVE"
"แย่มาก ไม่ตรงปก","NEGATIVE"
"โอเคนะ ใช้ได้","NEUTRAL"
```

### วิธีเทรนโมเดลแต่ละประเภท

#### 1. Logistic Regression (Model A - Baseline)

```bash
python Regress_train.py
```

**Output:**
- โมเดล: `models_regress/sentiment_model_*.joblib`
- Vectorizer: `models_regress/vectorizer_*.joblib`
- Evaluation: `results_regress/evaluation_*.png`

#### 2. Linear SVM

```bash
python Renear_train.py
```

#### 3. Random Forest

```bash
python "Random Forest_train.py"
```

#### 4. Naive Bayes

```bash
python naivebay.py
```

#### 5. LightGBM

```bash
python lightbgm.py
```

#### 6. Extra Trees

```bash
python extratree.py
```

### โครงสร้างการเทรน

แต่ละสคริปต์จะ:
1. โหลดและ preprocess ข้อมูล
2. Split train/test (80/20)
3. เทรนโมเดลด้วย TF-IDF vectorizer
4. ประเมินผล (Accuracy, F1-Score, Confusion Matrix)
5. บันทึกโมเดลพร้อม UID สำหรับ version control
6. บันทึก misclassified examples สำหรับการวิเคราะห์

---

## 📡 API Documentation

### Base URL

```
http://127.0.0.1:8000
```

### Endpoints

#### 1. **GET** `/` - หน้า Web UI หลัก

**Description**: แสดงหน้าเว็บสำหรับทดสอบโมเดล

**Response**: HTML page

---

#### 2. **POST** `/predict` - ทำนายด้วย Model A (Logistic Regression)

**Description**: ทำนายความรู้สึกด้วยโมเดล baseline

**Request Body:**
```json
{
  "text": "สินค้าดีมาก ประทับใจ 😊"
}
```

**Response:**
```json
{
  "label": "POSITIVE",
  "confidence": 0.95,
  "latency_ms": 12.34,
  "model": "sentiment_lr",
  "version": "TF-IDF + Logistic Regression (Linear, Probabilistic)",
  "important_words": ["ดีมาก", "ประทับใจ"],
  "word_sentiments": ["positive", "positive"]
}
```

**cURL Example:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"สินค้าดีมาก ประทับใจ"}'
```

---

#### 3. **POST** `/predict-ab` - เปรียบเทียบ Model A และ Model B

**Description**: ทำนายด้วย Model A และ Model B พร้อมกัน

**Request Body:**
```json
{
  "text": "สินค้าแย่มาก ผิดหวัง",
  "model_b_type": "linear"
}
```

**model_b_type options:**
- `"linear"` - Linear SVM
- `"rf"` - Random Forest
- `"nb"` - Naive Bayes
- `"lgbm"` - LightGBM
- `"et"` - Extra Trees

**Response:**
```json
{
  "model_a": {
    "label": "NEGATIVE",
    "confidence": 0.92,
    "latency_ms": 8.5,
    "model_name": "sentiment_lr",
    "version": "TF-IDF + Logistic Regression",
    "important_words": ["แย่มาก", "ผิดหวัง"],
    "word_sentiments": ["negative", "negative"]
  },
  "model_b": {
    "label": "NEGATIVE",
    "confidence": 0.94,
    "latency_ms": 12.3,
    "model_name": "Linear SVM",
    "version": "TF-IDF + Linear SVM (Max-Margin)",
    "important_words": ["แย่", "ผิดหวัง"],
    "word_sentiments": ["negative", "negative"]
  }
}
```

---

#### 4. **POST** `/feedback` - ส่ง Feedback

**Description**: บันทึก feedback จากผู้ใช้เพื่อปรับปรุงโมเดล

**Request Body:**
```json
{
  "text": "สินค้าดีมาก",
  "model": "model_a",
  "predicted_label": "POSITIVE",
  "feedback": "correct",
  "true_label": "POSITIVE",
  "confidence": 0.95,
  "model_name": "sentiment_lr",
  "timestamp": "2026-02-11T18:00:00"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Feedback recorded"
}
```

---

#### 5. **GET** `/errors` - ดูข้อผิดพลาด

**Description**: แสดงหน้ารายการข้อผิดพลาดจากการทำนาย

**Response**: HTML page แสดง 20 ข้อผิดพลาดล่าสุด

---

#### 6. **GET** `/health` - ตรวจสอบสถานะระบบ

**Description**: ตรวจสอบว่าระบบทำงานปกติหรือไม่

**Response:**
```json
{
  "status": "ok",
  "baseline_a": true,
  "available_models": ["linear", "rf", "nb", "lgbm", "et"]
}
```

---

#### 7. **GET** `/model/info` - ดูข้อมูลโมเดล

**Description**: แสดงข้อมูลโมเดลทั้งหมดที่โหลดไว้

**Response:**
```json
{
  "model_a": {
    "name": "sentiment_lr",
    "version": "TF-IDF + Logistic Regression",
    "file": "sentiment_model_20260210_173038_59628ab2.joblib"
  },
  "linear": {
    "name": "Linear SVM",
    "version": "TF-IDF + Linear SVM (Max-Margin)"
  },
  "rf": {
    "name": "Random Forest",
    "version": "TF-IDF + Random Forest"
  }
}
```

---

## 🚢 วิธี Deploy บน Render

Render เป็นแพลตฟอร์มที่ใช้งานง่ายสำหรับการ deploy web applications โดยมี Free Tier ให้ใช้งาน

### ขั้นตอนการ Deploy

#### 1. เตรียม Repository ให้พร้อม

ตรวจสอบว่าโปรเจคของคุณมีไฟล์เหล่านี้:
- ✅ `app.py` - FastAPI application
- ✅ `requirements.txt` - Python dependencies
- ✅ โฟลเดอร์ `models_*` - โมเดลที่เทรนแล้ว
- ✅ โฟลเดอร์ `templates/` และ `static/`

#### 2. Push โค้ดขึ้น GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

#### 3. สร้าง Web Service บน Render

1. ไปที่ [render.com](https://render.com) และสร้างบัญชี (ใช้ GitHub account)
2. คลิก **"New"** → **"Web Service"**
3. เชื่อมต่อ GitHub repository ของคุณ
4. ตั้งค่าดังนี้:

**Build Settings:**
- **Name**: `thai-sentiment-api` (หรือชื่อที่ต้องการ)
- **Region**: `Singapore` (ใกล้ที่สุดกับประเทศไทย)
- **Branch**: `main`
- **Root Directory**: (ว่างไว้)
- **Runtime**: `Python 3`
- **Build Command**: 
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  uvicorn app:app --host 0.0.0.0 --port $PORT
  ```

**Instance Type:**
- เลือก **Free** (512MB RAM, shared CPU)

> ⚠️ **หมายเหตุ**: Free tier จะหยุดทำงานหลังจากไม่มีการใช้งาน 15 นาที และจะ restart เมื่อมีคนเข้าใช้งานใหม่ (cold start ~30 วินาที)

#### 4. ตั้งค่า Environment Variables (ถ้าจำเป็น)

ไปที่ **Environment** tab และเพิ่ม:

```
PYTHON_VERSION=3.9.16
```

#### 5. คลิก "Create Web Service"

Render จะเริ่มทำการ build และ deploy โปรเจคของคุณ ใช้เวลาประมาณ 5-10 นาที

#### 6. เข้าถึงเว็บแอปของคุณ

เมื่อ deploy สำเร็จ คุณจะได้ URL แบบนี้:
```
https://thai-sentiment-api.onrender.com
```

### การอัปเดตโปรเจค

เมื่อคุณต้องการอัปเดตโค้ด:

```bash
git add .
git commit -m "Update code"
git push origin main
```

Render จะทำการ auto-deploy ใหม่โดยอัตโนมัติ!

### การจัดการโมเดลไฟล์ขนาดใหญ่

ถ้าโมเดลของคุณมีขนาดใหญ่มาก (>100MB) แนะนำให้:

**Option 1: ใช้ Git LFS (Large File Storage)**

```bash
# ติดตั้ง Git LFS
git lfs install

# Track โมเดลไฟล์
git lfs track "*.joblib"
git lfs track "*.pkl"

git add .gitattributes
git add models_*/*.joblib
git commit -m "Add models with Git LFS"
git push origin main
```

**Option 2: Download โมเดลตอน Build Time**

สร้างไฟล์ `download_models.py`:

```python
import requests
import os

MODEL_URLS = {
    "vectorizer": "https://your-storage-url/vectorizer.joblib",
    "model": "https://your-storage-url/model.joblib"
}

for name, url in MODEL_URLS.items():
    response = requests.get(url)
    with open(f"models/{name}.joblib", "wb") as f:
        f.write(response.content)
    print(f"Downloaded {name}")
```

แล้วแก้ **Build Command** ใน Render:
```bash
pip install -r requirements.txt && python download_models.py
```

### เพิ่มประสิทธิภาพสำหรับ Production

#### ใช้ Gunicorn (แนะนำ)

แก้ `requirements.txt` เพิ่ม:
```
gunicorn
```

แก้ **Start Command** ใน Render:
```bash
gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

#### ปรับแต่ง Workers

- **Free Plan**: ใช้ 1-2 workers
- **Paid Plan**: ใช้ 2-4 workers

### การ Monitor และ Logs

1. ไปที่ Render Dashboard → เลือก Web Service ของคุณ
2. คลิก **"Logs"** tab เพื่อดู real-time logs
3. คลิก **"Metrics"** tab เพื่อดู CPU/Memory usage

### Custom Domain (ถ้าต้องการ)

1. ไปที่ **Settings** tab
2. เลื่อนลงไปที่ **Custom Domains**
3. คลิก **"Add Custom Domain"**
4. ใส่ domain ของคุณ (เช่น `sentiment.yourdomain.com`)
5. ตั้งค่า DNS ตามที่ Render แนะนำ

### Troubleshooting สำหรับ Render

#### ปัญหา: Build ล้มเหลว
```
Error: Could not find a version that satisfies the requirement...
```
**แก้ไข**: ตรวจสอบ `requirements.txt` ว่ามี package version ที่ถูกต้อง

#### ปัญหา: Out of Memory
```
Error: Worker exited with code 137
```
**แก้ไข**: 
- ลด workers เหลือ 1
- Upgrade เป็น Paid Plan (512MB → 2GB+)
- ลดขนาดโมเดลโดยใช้ `max_features` ใน TF-IDF

#### ปัญหา: Cold Start ช้า
**แก้ไข**: 
- Upgrade เป็น Paid Plan (ไม่มี sleep mode)
- หรือใช้ cron job ping server ทุก 10 นาที

---

## 🤖 โมเดลที่รองรับ

### Model A (Baseline) - Logistic Regression

**เทคนิค**: TF-IDF + Logistic Regression  
**ข้อดี**:
- เร็วมาก (< 10ms)
- ใช้ RAM น้อย
- ให้ probability scores ที่เชื่อถือได้
- Explainable (ดูได้ว่าคำไหนมีน้ำหนักมาก)

**ข้อเสีย**:
- ไม่เข้าใจบริบทลึก
- จับ sarcasm ไม่ได้ดี

---

### Model B Options

#### 1. Linear SVM
**เทคนิค**: TF-IDF + Linear Support Vector Machine  
**ข้อดี**: ดีกับ high-dimensional data, effective กับ text classification  
**ข้อเสีย**: ช้ากว่า Logistic Regression เล็กน้อย

#### 2. Random Forest
**เทคนิค**: TF-IDF + Random Forest Classifier  
**ข้อดี**: จัดการ feature interaction ได้ดี, ป้องกัน overfitting  
**ข้อเสีย**: ช้ากว่า linear models, ใช้ RAM มากกว่า

#### 3. Naive Bayes
**เทคนิค**: TF-IDF + Multinomial Naive Bayes  
**ข้อดี**: เร็วมาก, ทำงานดีกับข้อมูลน้อย  
**ข้อเสีย**: สมมติฐาน independence ของคำไม่เป็นจริง

#### 4. LightGBM
**เทคนิค**: TF-IDF + LightGBM  
**ข้อดี**: เร็ว, ใช้ RAM น้อย, แม่นยำสูง  
**ข้อเสีย**: อาจ overfit ง่ายกับข้อมูลน้อย

#### 5. Extra Trees
**เทคนิค**: TF-IDF + Extra Trees Classifier  
**ข้อดี**: เร็วกว่า Random Forest, reduce variance  
**ข้อเสีย**: อาจมี bias สูงกว่า Random Forest

---

## 🔧 Troubleshooting

### ปัญหา: ImportError: No module named 'xxx'
**แก้ไข**: ติดตั้ง dependencies ใหม่
```bash
pip install -r requirements.txt
```

### ปัญหา: FileNotFoundError: model file not found
**แก้ไข**: รัน training script ก่อน
```bash
python Regress_train.py
```

### ปัญหา: uvicorn command not found
**แก้ไข**: ตรวจสอบว่า activate virtual environment แล้วหรือยัง
```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### ปัญหา: Port 8000 already in use
**แก้ไข**: เปลี่ยน port
```bash
uvicorn app:app --port 8080 --reload
```

### ปัญหา: Memory Error ระหว่างเทรนโมเดล
**แก้ไข**: ใช้ dataset ที่เล็กกว่า (5000 แทน 100k)
```python
# ในไฟล์ train script แก้ไข
df = pd.read_csv("data/1.synthetic_wisesight_like_thai_sentiment_5000.csv")
```

---

## 📊 Performance Benchmarks

| Model | Latency (avg) | Accuracy | F1-Score | RAM Usage |
|-------|---------------|----------|----------|-----------|
| Logistic Regression | 8ms | ~85% | ~0.83 | 150MB |
| Linear SVM | 12ms | ~86% | ~0.84 | 180MB |
| Random Forest | 45ms | ~84% | ~0.82 | 400MB |
| Naive Bayes | 5ms | ~80% | ~0.78 | 100MB |
| LightGBM | 25ms | ~86% | ~0.84 | 250MB |
| Extra Trees | 40ms | ~85% | ~0.83 | 380MB |

> ⚠️ ผลลัพธ์ข้างต้นเป็นเพียงตัวอย่าง ผลจริงขึ้นอยู่กับ hardware และ dataset

---

## 🖥️ การใช้งาน Web UI

### หน้าหลัก (/)

1. **กรอกข้อความ**: พิมพ์ข้อความที่ต้องการวิเคราะห์
2. **เลือกโหมด**:
   - ปิด A/B Testing: ใช้ Model A เพียงอย่างเดียว
   - เปิด A/B Testing: เปรียบเทียบ Model A กับ Model B
3. **เลือก Model B**: เลือกโมเดลที่ต้องการเปรียบเทียบ
4. **กดปุ่ม "วิเคราะห์"**: ดูผลลัพธ์

### ฟีเจอร์เพิ่มเติม

- **โหลดตัวอย่างข้อความ**: สุ่มข้อความตัวอย่างเพื่อทดสอบ
- **ดูตัวอย่างข้อผิดพลาด**: ดูรายการข้อความที่โมเดลทำนายผิด
- **Feedback System**: 
  - กด 👍 ถ้าผลลัพธ์ถูกต้อง
  - กด 👎 ถ้าผลลัพธ์ผิด พร้อมระบุคำตอบที่ถูกต้อง

### ผลลัพธ์ที่แสดง

- **Label**: ประเภทความรู้สึก (POSITIVE/NEGATIVE/NEUTRAL)
- **Confidence**: ความมั่นใจของโมเดล (0.00 - 1.00)
- **Latency**: เวลาที่ใช้ในการทำนาย (milliseconds)
- **Important Words**: คำที่มีอิทธิพลต่อการทำนาย พร้อมสี:
  - 🟢 เขียว = คำบวก
  - 🔴 แดง = คำลบ
  - 🟡 เหลือง = คำกลาง

---

## 📝 License

MIT License - สามารถใช้งานได้อย่างอิสระ

---

## 👨‍💻 Author

**Phurin (Phurin123)**

GitHub: [https://github.com/Phurin123](https://github.com/Phurin123)

---

## 🙏 Acknowledgments

- **Wisesight Sentiment Corpus** - สำหรับ training data concept
- **pythainlp** - Thai NLP tools
- **FastAPI** - Modern web framework
- **scikit-learn** - Machine learning library

---

## 📮 Contact & Support

หากพบปัญหาหรือมีคำถาม:
- เปิด Issue ใน [GitHub Repository](https://github.com/Phurin123/Thai-Sentiment-Analysis-System-Using-TF-IDF)
- ติดต่อผ่าน GitHub Profile

---

**Made with ❤️ for Thai NLP Community**
