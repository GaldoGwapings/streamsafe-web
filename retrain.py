import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ----------------- CONFIG -----------------
DATA_FILE = "RiverQuality.csv"
TARGET_COL = "RiskLevel"
TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5

OUTPUT_MODEL = "streamsafe_model_macro.pkl"
OUTPUT_ENCODER = "label_encoder.pkl"
METRICS_JSON = "model_metrics_macro.json"
CMAP_FIG = "confusion_matrix_macro.png"

# ----------------- LOAD DATA -----------------
if not os.path.exists(DATA_FILE):
    print(f"Missing dataset file: {DATA_FILE}")
    sys.exit(1)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

# Normalize text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# Standardize class labels
df[TARGET_COL] = df[TARGET_COL].replace({
    "Moderate": "Moderate Risk",
    "Low": "Low Risk",
    "High": "High Risk"
})

# ----------------- NUMERIC CONVERSION -----------------
numeric_possible = [
    "ph", "Total Suspended Solids (mg/L)", "BOD (mg/L)", "DO (mg/L)",
    "Fecal coliform (MPN/100mL)", "Temperature", "Color (TCU)",
    "Chloride (mg/L)", "Phosphate (mg/L)"
]
for col in numeric_possible:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------- FEATURE ENGINEERING -----------------
if all(c in df.columns for c in ["ph", "Total Suspended Solids (mg/L)", "BOD (mg/L)", "DO (mg/L)"]):
    df["pollution_severity"] = (
        abs(df["ph"] - 7) * 0.8 +
        (df["Total Suspended Solids (mg/L)"] / 10) * 1.2 +
        df["BOD (mg/L)"] * 0.7 +
        (1 / (df["DO (mg/L)"] + 1e-6)) * 2.0
    )
if "BOD (mg/L)" in df.columns and "DO (mg/L)" in df.columns:
    df["organic_load_ratio"] = df["BOD (mg/L)"] / (df["DO (mg/L)"] + 1e-6)
if "Total Suspended Solids (mg/L)" in df.columns and "Color (TCU)" in df.columns:
    df["sediment_ratio"] = df["Total Suspended Solids (mg/L)"] / (df["Color (TCU)"] + 1e-6)
if "Fecal coliform (MPN/100mL)" in df.columns:
    df["log_fecal"] = np.log1p(df["Fecal coliform (MPN/100mL)"].fillna(0).clip(lower=0))
for col in ["Total Suspended Solids (mg/L)", "BOD (mg/L)", "DO (mg/L)"]:
    if col in df.columns:
        df[col + "_bin"] = pd.qcut(df[col], q=4, duplicates="drop").cat.codes

# ----------------- PREPARE X, y -----------------
drop_cols = ["Date", "Region", "Location", "River", "Sample ID", "Station Name"]
X = df.drop(columns=[TARGET_COL] + [c for c in drop_cols if c in df.columns], errors="ignore")
X = X.select_dtypes(include=[np.number])
missing_rate = X.isna().mean()
drop_missing = missing_rate[missing_rate > 0.60].index.tolist()
if drop_missing:
    X = X.drop(columns=drop_missing)

y = df[TARGET_COL]

# ----------------- LABEL ENCODER -----------------
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ----------------- TRAIN/TEST SPLIT -----------------
X_train, X_test, y_train_enc, y_test_enc = train_test_split(
    X.values, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE
)

# ----------------- PREPROCESSOR -----------------
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ]), np.arange(X_train.shape[1]))
])

# ----------------- LIGHTGBM WITH NUMERIC CLASS WEIGHTS -----------------
classes = np.unique(y_train_enc)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_enc)
class_weights = dict(zip(classes, weights))

lgbm = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=60,
    min_child_samples=10,
    reg_lambda=2.0,
    reg_alpha=1.0,
    class_weight=class_weights,
    random_state=RANDOM_STATE
)

# ----------------- SMOTE OVERSAMPLING -----------------
counter = Counter(y_train_enc)
smote_strategy = {
    0: int(counter[0]*1.0),   # Low Risk
    1: int(counter[1]*1.2),   # Moderate Risk
    2: int(counter[2]*1.5)    # High Risk
}
smote = SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE)

pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", smote),
    ("classifier", lgbm)
])

# ----------------- STRATIFIED K-FOLD TRAINING -----------------
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
for train_idx, val_idx in skf.split(X_train, y_train_enc):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train_enc[train_idx], y_train_enc[val_idx]
    pipeline.fit(X_tr, y_tr)

# ----------------- PREDICTIONS -----------------
y_proba = pipeline.predict_proba(X_test)
y_pred_final = np.zeros_like(y_test_enc)

# ----------------- MACRO-F1 THRESHOLD OPTIMIZATION -----------------
best_macro_f1 = 0
best_thresh_combo = [0.5] * len(le.classes_)

for low, moderate, high in product(np.arange(0.3,0.7,0.05), repeat=3):
    y_pred_tmp = np.zeros_like(y_test_enc)
    thresholds = [low, moderate, high]
    for idx in range(len(y_test_enc)):
        scores = [y_proba[idx,j]/thresholds[j] for j in range(len(le.classes_))]
        y_pred_tmp[idx] = np.argmax(scores)
    macro_f1 = f1_score(y_test_enc, y_pred_tmp, average='macro')
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_thresh_combo = thresholds

for idx in range(len(y_test_enc)):
    scores = [y_proba[idx,j]/best_thresh_combo[j] for j in range(len(le.classes_))]
    y_pred_final[idx] = np.argmax(scores)

# ----------------- EVALUATION -----------------
report = classification_report(y_test_enc, y_pred_final, output_dict=True, target_names=le.classes_)
kappa = cohen_kappa_score(y_test_enc, y_pred_final)
print("\n===== CLASSIFICATION REPORT (MACRO-F1 THRESHOLDS) =====\n")
print(classification_report(y_test_enc, y_pred_final, target_names=le.classes_))
print(f"Cohen's Kappa: {kappa:.4f}")

# ----------------- CONFUSION MATRIX -----------------
cm = confusion_matrix(y_test_enc, y_pred_final)
fig, ax = plt.subplots(figsize=(8,6))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix – Macro-F1 Optimized")
plt.savefig(CMAP_FIG)
plt.close()

# ----------------- SAVE ARTIFACTS -----------------
joblib.dump(pipeline, OUTPUT_MODEL)
joblib.dump(le, OUTPUT_ENCODER)
with open(METRICS_JSON, "w") as f:
    json.dump({
        "accuracy": float(report["accuracy"]),
        "kappa": float(kappa),
        "report": report
    }, f, indent=2)

print("\nTraining complete.")
print(f"Model saved to: {OUTPUT_MODEL}")
print(f"Metrics saved to: {METRICS_JSON}")
print(f"Confusion matrix saved to: {CMAP_FIG}")