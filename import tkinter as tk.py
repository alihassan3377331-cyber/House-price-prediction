import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import threading

# ─────────────────────────────────────────────
#  Global state
# ─────────────────────────────────────────────
df_global = None
model_global = None
encoders = {}
feature_cols = ['area_type', 'availability', 'location', 'size',
                'society', 'total_sqft', 'bath', 'balcony']

# ─────────────────────────────────────────────
#  Data processing  (same logic as notebook)
# ─────────────────────────────────────────────
def preprocess(df):
    house = df.copy()

    # Label-encode categorical columns
    cat_cols = ['area_type', 'availability', 'location', 'size', 'society']
    for col in cat_cols:
        le = LabelEncoder()
        house[col] = le.fit_transform(house[col].astype(str))
        encoders[col] = le

    # Convert total_sqft to numeric (handle ranges like "1000-1200")
    def convert_sqft(val):
        try:
            parts = str(val).split('-')
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
            return float(val)
        except:
            return np.nan

    house['total_sqft'] = house['total_sqft'].apply(convert_sqft)

    # Fill missing values
    house['bath'].fillna(house['bath'].median(), inplace=True)
    house['balcony'].fillna(house['balcony'].median(), inplace=True)
    house['total_sqft'].fillna(house['total_sqft'].median(), inplace=True)
    house['price'].fillna(house['price'].median(), inplace=True)

    return house

# ─────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────
class HousePriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏠 House Price Prediction")
        self.root.geometry("900x680")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        self.model_var = tk.StringVar(value="Linear Regression")
        self.file_path = tk.StringVar(value="No file selected")
        self.status_var = tk.StringVar(value="Ready — please load a CSV file first.")

        self._build_ui()

    # ── UI Builder ──────────────────────────────
    def _build_ui(self):
        BG   = "#1e1e2e"
        CARD = "#2a2a3e"
        ACC  = "#7c3aed"
        ACC2 = "#a855f7"
        TXT  = "#e2e8f0"
        SUB  = "#94a3b8"
        GRN  = "#22c55e"
        RED  = "#ef4444"

        # ── Header ──
        hdr = tk.Frame(self.root, bg=ACC, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🏠  House Price Prediction System",
                 font=("Segoe UI", 18, "bold"), bg=ACC, fg="white").pack()
        tk.Label(hdr, text="Bangalore Real Estate Dataset  |  ML-Powered",
                 font=("Segoe UI", 10), bg=ACC, fg="#ddd6fe").pack()

        # ── Main body ──
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=14)

        # Left column
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Right column
        right = tk.Frame(body, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        # ── CARD: Load Data ──
        self._card(left, "📂  Step 1 — Load CSV Data", BG, CARD, TXT, SUB, ACC2, GRN, RED)

        # ── CARD: Train Model ──
        self._train_card(left, BG, CARD, TXT, SUB, ACC, ACC2, GRN, RED)

        # ── CARD: Predict ──
        self._predict_card(right, BG, CARD, TXT, SUB, ACC, ACC2, GRN, RED)

        # ── Status bar ──
        sb = tk.Frame(self.root, bg="#12121f", pady=6)
        sb.pack(fill="x", side="bottom")
        tk.Label(sb, textvariable=self.status_var,
                 font=("Segoe UI", 9), bg="#12121f", fg=SUB,
                 anchor="w", padx=12).pack(fill="x")

    def _card(self, parent, title, BG, CARD, TXT, SUB, ACC2, GRN, RED):
        """Load data card"""
        frame = tk.Frame(parent, bg=CARD, bd=0, relief="flat")
        frame.pack(fill="x", pady=(0, 12))
        tk.Frame(frame, bg=ACC2, height=3).pack(fill="x")
        inner = tk.Frame(frame, bg=CARD, padx=16, pady=14)
        inner.pack(fill="x")

        tk.Label(inner, text=title, font=("Segoe UI", 11, "bold"),
                 bg=CARD, fg=TXT).pack(anchor="w")
        tk.Label(inner, text="Select your House_Price_Data.csv file",
                 font=("Segoe UI", 9), bg=CARD, fg=SUB).pack(anchor="w", pady=(2, 8))

        file_row = tk.Frame(inner, bg=CARD)
        file_row.pack(fill="x")
        tk.Label(file_row, textvariable=self.file_path, font=("Segoe UI", 8),
                 bg="#12121f", fg=SUB, anchor="w", padx=8, pady=6,
                 relief="flat", width=34).pack(side="left", fill="x", expand=True)
        tk.Button(file_row, text="Browse", font=("Segoe UI", 9, "bold"),
                  bg=ACC2, fg="white", relief="flat", padx=12, cursor="hand2",
                  command=self._load_file).pack(side="left", padx=(6, 0))

        self.data_info = tk.Label(inner, text="", font=("Segoe UI", 9),
                                  bg=CARD, fg=GRN, anchor="w")
        self.data_info.pack(anchor="w", pady=(6, 0))

    def _train_card(self, parent, BG, CARD, TXT, SUB, ACC, ACC2, GRN, RED):
        """Train model card"""
        frame = tk.Frame(parent, bg=CARD, bd=0)
        frame.pack(fill="x", pady=(0, 12))
        tk.Frame(frame, bg=ACC, height=3).pack(fill="x")
        inner = tk.Frame(frame, bg=CARD, padx=16, pady=14)
        inner.pack(fill="x")

        tk.Label(inner, text="🤖  Step 2 — Train Model",
                 font=("Segoe UI", 11, "bold"), bg=CARD, fg=TXT).pack(anchor="w")
        tk.Label(inner, text="Choose algorithm and train on the dataset",
                 font=("Segoe UI", 9), bg=CARD, fg=SUB).pack(anchor="w", pady=(2, 10))

        # Model selector
        models = ["Linear Regression", "Decision Tree", "KNN (k=1)"]
        sel_frame = tk.Frame(inner, bg=CARD)
        sel_frame.pack(fill="x", pady=(0, 10))
        tk.Label(sel_frame, text="Algorithm:", font=("Segoe UI", 9),
                 bg=CARD, fg=SUB).pack(side="left")
        combo = ttk.Combobox(sel_frame, textvariable=self.model_var,
                             values=models, state="readonly",
                             font=("Segoe UI", 9), width=22)
        combo.pack(side="left", padx=(8, 0))

        self.train_btn = tk.Button(inner, text="⚡  Train Model",
                                   font=("Segoe UI", 10, "bold"),
                                   bg=ACC, fg="white", relief="flat",
                                   pady=8, cursor="hand2",
                                   command=self._train_thread)
        self.train_btn.pack(fill="x", pady=(4, 8))

        # Metrics display
        mf = tk.Frame(inner, bg="#12121f", padx=12, pady=10)
        mf.pack(fill="x")
        self.mae_var  = tk.StringVar(value="MAE  :  —")
        self.r2_var   = tk.StringVar(value="R²   :  —")
        self.acc_var  = tk.StringVar(value="Accuracy  :  —")
        for var, clr in [(self.mae_var, SUB), (self.r2_var, GRN), (self.acc_var, ACC2)]:
            tk.Label(mf, textvariable=var, font=("Courier New", 10),
                     bg="#12121f", fg=clr).pack(anchor="w")

    def _predict_card(self, parent, BG, CARD, TXT, SUB, ACC, ACC2, GRN, RED):
        """Prediction card"""
        frame = tk.Frame(parent, bg=CARD, bd=0)
        frame.pack(fill="both", expand=True)
        tk.Frame(frame, bg=GRN, height=3).pack(fill="x")
        inner = tk.Frame(frame, bg=CARD, padx=16, pady=14)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text="🔮  Step 3 — Predict Price",
                 font=("Segoe UI", 11, "bold"), bg=CARD, fg=TXT).pack(anchor="w")
        tk.Label(inner, text="Enter property details to get price estimate",
                 font=("Segoe UI", 9), bg=CARD, fg=SUB).pack(anchor="w", pady=(2, 10))

        fields_frame = tk.Frame(inner, bg=CARD)
        fields_frame.pack(fill="x")

        self.inputs = {}
        fields = [
            ("Area Type",    "area_type",   ["Super built-up  Area", "Built-up  Area",
                                              "Plot  Area", "Carpet  Area"]),
            ("Availability", "availability",["Ready To Move", "19-Dec", "18-Jun",
                                              "18-Dec", "19-Jun", "Other"]),
            ("Location",     "location",    None),
            ("Size",         "size",        ["1 BHK", "2 BHK", "3 BHK", "4 BHK",
                                              "5 BHK", "2 Bedroom", "3 Bedroom",
                                              "4 Bedroom", "5 Bedroom", "6 Bedroom"]),
            ("Society",      "society",     None),
            ("Total Sqft",   "total_sqft",  None),
            ("Bathrooms",    "bath",        ["1", "2", "3", "4", "5", "6"]),
            ("Balconies",    "balcony",     ["0", "1", "2", "3"]),
        ]

        for i, (label, key, options) in enumerate(fields):
            row = tk.Frame(fields_frame, bg=CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label + ":", font=("Segoe UI", 9),
                     bg=CARD, fg=SUB, width=13, anchor="w").pack(side="left")
            if options:
                var = tk.StringVar(value=options[0])
                widget = ttk.Combobox(row, textvariable=var, values=options,
                                      font=("Segoe UI", 9), width=22, state="readonly")
            else:
                var = tk.StringVar()
                widget = tk.Entry(row, textvariable=var, font=("Segoe UI", 9),
                                  bg="#12121f", fg="white", insertbackground="white",
                                  relief="flat", width=25)
                if key == "total_sqft": var.set("1200")
                if key == "location":   var.set("Whitefield")
                if key == "society":    var.set("Unknown")
            widget.pack(side="left", padx=(4, 0))
            self.inputs[key] = var

        self.predict_btn = tk.Button(inner, text="💰  Predict Price",
                                     font=("Segoe UI", 11, "bold"),
                                     bg=GRN, fg="white", relief="flat",
                                     pady=10, cursor="hand2",
                                     command=self._predict)
        self.predict_btn.pack(fill="x", pady=(14, 8))

        # Result display
        result_frame = tk.Frame(inner, bg="#12121f", padx=14, pady=14)
        result_frame.pack(fill="x")
        tk.Label(result_frame, text="Estimated Price", font=("Segoe UI", 9),
                 bg="#12121f", fg=SUB).pack()
        self.result_var = tk.StringVar(value="— Lakhs")
        tk.Label(result_frame, textvariable=self.result_var,
                 font=("Segoe UI", 22, "bold"), bg="#12121f", fg=ACC2).pack()
        self.result_note = tk.Label(result_frame, text="",
                                    font=("Segoe UI", 8), bg="#12121f", fg=SUB)
        self.result_note.pack()

    # ── Helpers ──────────────────────────────────
    def _set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _load_file(self):
        global df_global, encoders
        path = filedialog.askopenfilename(
            title="Select House Price CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self._set_status("Loading CSV…")
            df = pd.read_csv(path)
            required = ['area_type', 'availability', 'location', 'size',
                        'society', 'total_sqft', 'bath', 'balcony', 'price']
            missing = [c for c in required if c not in df.columns]
            if missing:
                messagebox.showerror("Column Error",
                                     f"Missing columns: {', '.join(missing)}")
                return
            df_global = df
            encoders = {}
            short = path.split("/")[-1] if "/" in path else path.split("\\")[-1]
            self.file_path.set(short)
            self.data_info.config(
                text=f"✔  {len(df):,} rows × {len(df.columns)} columns loaded"
            )
            self._set_status(f"Data loaded: {len(df):,} records. Now train a model.")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self._set_status("Error loading file.")

    def _train_thread(self):
        t = threading.Thread(target=self._train, daemon=True)
        t.start()

    def _train(self):
        global df_global, model_global, encoders
        if df_global is None:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return
        try:
            self.train_btn.config(state="disabled", text="Training…")
            self._set_status("Preprocessing data…")

            processed = preprocess(df_global)
            X = processed[feature_cols]
            y = processed['price']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            algo = self.model_var.get()
            self._set_status(f"Training {algo}…")

            if algo == "Linear Regression":
                model = LinearRegression()
            elif algo == "Decision Tree":
                model = DecisionTreeRegressor(random_state=42)
            else:
                model = KNeighborsRegressor(n_neighbors=1)

            model.fit(X_train, y_train)
            model_global = model

            y_pred = model.predict(X_test)
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            acc  = max(0, r2) * 100

            self.mae_var.set(f"MAE  :  {mae:.2f} Lakhs")
            self.r2_var.set(f"R²   :  {r2:.4f}")
            self.acc_var.set(f"Accuracy  :  {acc:.1f}%")

            self._set_status(
                f"✔ {algo} trained | MAE={mae:.2f} | R²={r2:.4f} | "
                f"Train={len(X_train):,}  Test={len(X_test):,}"
            )
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self._set_status("Training failed.")
        finally:
            self.train_btn.config(state="normal", text="⚡  Train Model")

    def _predict(self):
        global model_global, encoders
        if model_global is None:
            messagebox.showwarning("No Model", "Please train a model first.")
            return
        try:
            vals = {}
            for key, var in self.inputs.items():
                vals[key] = var.get().strip()
                if not vals[key]:
                    messagebox.showwarning("Input Missing",
                                           f"Please fill in: {key}")
                    return

            # Encode categoricals using fitted encoders
            row = {}
            cat_cols = ['area_type', 'availability', 'location', 'size', 'society']
            for col in cat_cols:
                le = encoders.get(col)
                val = vals[col]
                if le is not None:
                    classes = list(le.classes_)
                    # handle unseen labels gracefully
                    if val not in classes:
                        # use most-frequent class index
                        row[col] = 0
                    else:
                        row[col] = le.transform([val])[0]
                else:
                    row[col] = 0

            row['total_sqft'] = float(vals['total_sqft'])
            row['bath']       = float(vals['bath'])
            row['balcony']    = float(vals['balcony'])

            X_new = pd.DataFrame([row])[feature_cols]
            price = model_global.predict(X_new)[0]

            self.result_var.set(f"₹ {price:.2f}  Lakhs")
            cr = price * 100_000       # 1 lakh = 100,000
            self.result_note.config(
                text=f"≈ ₹ {cr:,.0f}  |  Model: {self.model_var.get()}"
            )
            self._set_status(f"Prediction complete: ₹ {price:.2f} Lakhs")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self._set_status("Prediction failed.")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()

    # Style the ttk combobox
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TCombobox",
                    fieldbackground="#12121f",
                    background="#2a2a3e",
                    foreground="white",
                    arrowcolor="white",
                    selectbackground="#7c3aed")

    app = HousePriceApp(root)
    root.mainloop()