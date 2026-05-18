# 🚗 Automatikus Rendszámtábla Felismerő (ANPR)

Python-alapú automatikus rendszámtábla-felismerő rendszer, amely képes gépjárművekről készült képeken:

- 📍 rendszámtábla detektálására
- 🛠️ kép előfeldolgozására
- 🔠 karakterek szegmentálására
- 🤖 OCR-alapú karakterfelismerésre gépi tanulással

A projekt tartalmaz:

- egy **CLI alkalmazást** gyors teszteléshez
- egy **interaktív Streamlit webes felületet**
- YOLOv8-alapú detektálást fallback megoldással
- HOG + SVM alapú OCR rendszert

---

# ✨ Funkciók

## 📌 1. Rendszámtábla detektálás

A rendszer első lépésben lokalizálja a rendszámtáblát:

- **YOLOv8 objektumdetektáló modell**
- automatikus fallback:
  - Canny edge detection
  - kontúralapú keresés
  - méretarány-szűrés

📄 `plate_detector.py`

---

## 🖼️ 2. Kép-előfeldolgozás

Az OCR pontosságának javítása érdekében:

- 📐 Dőlés-korrekció (Deskew)
- 🌫️ Gauss-szűrés
- ⚫ Szürkeárnyalatos konverzió
- 🌗 CLAHE kontrasztnövelés
- 🧠 Niblack adaptív binarizálás
- 🧹 Morfológiai zajszűrés

📄 `image_pipeline.py`

---

## 🔠 3. Karakterszegmentálás

Kontúralapú karakterkivágás:

- Non-Maximum Suppression (NMS)
- geometriai szűrés

📄 `char_segmenter.py`

---

## 🤖 4. OCR – Karakterfelismerés

A rendszer:

1. HOG feature extraction-t használ
2. majd egy GridSearchCV-vel optimalizált SVM modellt alkalmaz

📄 `hog_extractor.py`  
📄 `knn_classifier.py`

> A modul neve `knn_classifier.py`, azonban valójában `SVC` modellt használ.

---

# 🧠 Technológiák

- Python 3.8+
- OpenCV
- Scikit-learn
- Ultralytics YOLOv8
- Streamlit
- NumPy
- Matplotlib

---

# 📦 Telepítés

## 1️⃣ Repository klónozása

```bash
git clone <repository-url>
cd <repository-folder>
```

## 2️⃣ Függőségek telepítése

```bash
pip install -r requirements.txt
```

---

# 🚀 Használat

## 🎓 Modell betanítása

A karakterfelismerő modell szintetikus adatokon tanítható:

```bash
python train_classifier.py
```

A script:

- automatikusan generál karaktereket
- különböző torzításokat alkalmaz
- betanítja az SVM modellt

### Generált modellek

```text
data/model/svm_model.pkl
data/model/label_encoder.pkl
```

---

# 💻 CLI használat

Teljes pipeline futtatása egy képen:

```bash
python main.py --image data/images/auto.jpg
```

## Elérhető opciók

| Argumentum | Leírás |
|---|---|
| `--image` | Bemeneti kép útvonala *(kötelező)* |
| `--no-visualize` | Vizualizáció kikapcsolása |
| `--model` | Egyedi YOLOv8 `best.pt` modell |

### Példa

```bash
python main.py --image car.jpg --no-visualize
```

---

# 🌐 Streamlit Webes Felület

Interaktív vizualizáció és képfeltöltés:

```bash
streamlit run streamlit_app.py
```

A felületen megtekinthető:

- CLAHE eredmény
- binarizált kép
- karakterkontúrok
- OCR predikciók
- teljes pipeline

---

# 📂 Projektstruktúra

```text
├── data/
│   ├── fonts/                # Betűtípusok a tanításhoz
│   ├── images/               # Tesztképek
│   └── model/                # Betanított modellek
│
├── src/
│   ├── detection/
│   │   └── plate_detector.py
│   │
│   ├── preprocessing/
│   │   └── image_pipeline.py
│   │
│   ├── segmentation/
│   │   └── char_segmenter.py
│   │
│   ├── recognition/
│   │   ├── hog_extractor.py
│   │   └── knn_classifier.py
│   │
│   └── visualization/
│       └── visualizer.py
│
├── main.py
├── train_classifier.py
├── streamlit_app.py
├── requirements.txt
└── README.md
