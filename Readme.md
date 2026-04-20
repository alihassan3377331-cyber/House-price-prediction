# 🏠 House Price Prediction System

> Bangalore real estate dataset par machine learning use karke ghar ki price predict karne wala desktop application.

---

## 📁 Project Files

```
House-Price-Prediction/
│
├── House_Price_Data.csv              ← Dataset (13,320 records)
├── House_Price_Data_csv.ipynb        ← Jupyter Notebook (analysis + model training)
├── house_price_gui.py                ← Tkinter Desktop GUI Application
└── README.md                         ← Yeh file
```

---

## 📊 Dataset — `House_Price_Data.csv`

Dataset mein Bangalore ke **13,320 ghar** ka data hai.

| Column | Type | Description |
|---|---|---|
| `area_type` | Categorical | Super built-up / Built-up / Plot / Carpet Area |
| `availability` | Categorical | Ready To Move ya possession date |
| `location` | Categorical | Bangalore ka area (1,306 unique locations) |
| `size` | Categorical | BHK ya Bedroom count (1 BHK – 6 Bedroom) |
| `society` | Categorical | Society / building ka naam |
| `total_sqft` | Numeric | Ghar ka total area (square feet) |
| `bath` | Numeric | Bathrooms ki tadaad |
| `balcony` | Numeric | Balconies ki tadaad |
| `price` | Numeric | **Target** — Price in Lakhs (₹) |

**Price Range:** ₹8 Lakhs → ₹3,600 Lakhs

---

## 📓 Jupyter Notebook — `House_Price_Data_csv.ipynb`

Notebook mein yeh steps hain:

**1. Data Loading**
```python
house = pd.read_csv('House_Price_Data.csv')
```

**2. Data Preprocessing**
- Categorical columns ka Label Encoding (area_type, location, size, etc.)
- `total_sqft` range values handle karna (e.g., `"1000-1200"` → `1100.0`)
- Missing values ko median se fill karna (bath, balcony, price)

**3. Feature Engineering**
- Features (X): `area_type, availability, location, size, society, total_sqft, bath, balcony`
- Target (y): `price`
- Train/Test Split: **80% / 20%**

**4. Models Train Kiye**

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression | ~46 Lakhs | ~0.00 |
| Decision Tree | varies | varies |
| KNN (k=1) | 46.22 Lakhs | -0.007 |

**5. Evaluation Metrics**
- Mean Absolute Error (MAE)
- R² Score
- Accuracy (R² ko percentage mein)

---

## 🖥️ Desktop GUI — `house_price_gui.py`

Tkinter se bani **dark theme** wali modern GUI application.

### Screenshots Overview

```
┌──────────────────────────────────────────────────┐
│   🏠  House Price Prediction System              │
│       Bangalore Real Estate Dataset | ML-Powered │
├─────────────────────┬────────────────────────────┤
│  📂 Step 1          │  🔮 Step 3 — Predict Price │
│  Load CSV Data      │                            │
│                     │  Area Type:  [dropdown]    │
│  📁 [file name]     │  Availability:[dropdown]   │
│  [Browse Button]    │  Location:   [text field]  │
│                     │  Size:       [dropdown]    │
│  🤖 Step 2          │  Society:    [text field]  │
│  Train Model        │  Total Sqft: [text field]  │
│                     │  Bathrooms:  [dropdown]    │
│  Algorithm:[combo]  │  Balconies:  [dropdown]    │
│  [⚡ Train Model]   │                            │
│                     │  [💰 Predict Price]        │
│  MAE  : —           │                            │
│  R²   : —           │  Estimated Price           │
│  Accuracy: —        │  ₹ 85.00  Lakhs            │
├─────────────────────┴────────────────────────────┤
│  Status: Ready — please load a CSV file first.   │
└──────────────────────────────────────────────────┘
```

### GUI Features
- **3-Step Workflow** — Load → Train → Predict
- **3 ML Algorithms** — Linear Regression, Decision Tree, KNN
- **Live Metrics** — MAE, R², Accuracy training ke baad dikhtay hain
- **Price Result** — Lakhs aur exact rupees dono mein
- **Dark Theme** — Modern purple/dark UI
- **Threading** — Training ke dauran UI freeze nahi hota
- **Error Handling** — Missing values aur unseen locations handle karta hai

---

## ⚙️ Installation — Kaise Setup Karein

### Step 1 — Python Check Karein
```bash
python --version
# Python 3.8 ya usse upar hona chahiye
```

### Step 2 — Required Libraries Install Karein
```bash
pip install pandas numpy scikit-learn
```

> **Note:** `tkinter` Python ke saath already aata hai, alag install karne ki zaroorat nahi.

### Step 3 — Files Aik Folder Mein Rakhein
```
MyProject/
├── House_Price_Data.csv
└── house_price_gui.py
```

---

## ▶️ Application Kaise Chalayein

### VS Code mein:
1. `house_price_gui.py` file VS Code mein open karein
2. Terminal mein jaayein (`Ctrl + ~`)
3. Yeh command chalayein:
```bash
python house_price_gui.py
```

### Direct command se:
```bash
# Windows
python house_price_gui.py

# Mac / Linux
python3 house_price_gui.py
```

---

## 📖 Application Use Karne Ka Tarika

### Step 1 — Data Load Karein
1. **Browse** button dabao
2. `House_Price_Data.csv` file select karo
3. Green text mein confirm hoga: `✔ 13,320 rows × 9 columns loaded`

### Step 2 — Model Train Karein
1. Dropdown se algorithm choose karo:
   - `Linear Regression` — Simple aur fast
   - `Decision Tree` — Complex patterns samajhta hai
   - `KNN (k=1)` — Nearest neighbor method
2. **⚡ Train Model** button dabao
3. Training complete hone par metrics dikhenge:
   - **MAE** — Average error kitna hai (Lakhs mein)
   - **R²** — Model ki accuracy score
   - **Accuracy %** — Percentage mein accuracy

### Step 3 — Price Predict Karein
| Field | Example |
|---|---|
| Area Type | Super built-up Area |
| Availability | Ready To Move |
| Location | Whitefield |
| Size | 3 BHK |
| Society | Unknown (koi bhi likhein) |
| Total Sqft | 1500 |
| Bathrooms | 3 |
| Balconies | 2 |

4. **💰 Predict Price** dabao
5. Result aayega: `₹ 95.50 Lakhs` (aur exact rupees)

---

## 🔧 Technical Details

### Data Preprocessing Pipeline
```
Raw CSV
   ↓
Label Encoding (area_type, availability, location, size, society)
   ↓
total_sqft range convert ("1000-1200" → 1100.0)
   ↓
Missing values → Median fill
   ↓
Train/Test Split (80/20, random_state=42)
   ↓
Model Training
```

### Machine Learning Models

```python
# Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)

# KNN
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=1)
```

### Libraries Used

| Library | Version | Use |
|---|---|---|
| `pandas` | latest | Data load aur processing |
| `numpy` | latest | Numerical operations |
| `scikit-learn` | latest | ML models aur metrics |
| `tkinter` | built-in | GUI application |
| `threading` | built-in | Background training |

---

## ❓ Common Errors aur Solutions

| Error | Wajah | Solution |
|---|---|---|
| `ModuleNotFoundError: pandas` | Library install nahi | `pip install pandas numpy scikit-learn` chalao |
| `Column Error: Missing columns` | Galat CSV file | Sirf `House_Price_Data.csv` use karo |
| `No Model` warning | Train nahi kiya | Pehle Step 2 mein train karo |
| GUI nahi khuli | Python path issue | `python3` use karo `python` ki jagah |

---

## 👨‍💻 Project Structure Summary

```
Notebook (.ipynb)     →  Data analysis, exploration, model testing
GUI (.py)             →  Same logic, user-friendly desktop interface
CSV (.csv)            →  Bangalore real estate training data
README.md             →  Documentation (yeh file)
```

---

*Dataset source: Bangalore House Price Data — commonly used ML practice dataset.*
