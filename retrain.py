import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from itertools import cycle

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             classification_report, cohen_kappa_score, roc_curve, auc)

# Model Import
from imblearn.ensemble import BalancedRandomForestClassifier

# --- 1. Load and Clean Data ---
try:
    df = pd.read_csv('RiverQuality.csv')
    df.columns = df.columns.str.strip() 
    
    target_col = 'RiskLevel'
    if target_col in df.columns:
        df[target_col] = df[target_col].str.strip().replace('Moderate', 'Moderate Risk')
    else:
        print(f"❌ Error: Column '{target_col}' not found.")
        exit()
    print("✅ Loaded Data")
except FileNotFoundError:
    print("❌ Error: Could not find 'RiverQuality.csv'.")
    exit()

# --- 2. Cleaning & Feature Engineering ---
print("🧪 Engineering Features & Treating Outliers...")

# A. Imputation & Outlier Treatment (Clipping Extreme Values)
# We clip values to the 1st and 99th percentile to handle outliers before scaling
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

# B. Create 'Pollution_Index' (Feature Engineering)
tss_col = 'Total Suspended Solids (mg/L)' 
if tss_col in df.columns:
    df[tss_col] = df[tss_col].fillna(df[tss_col].median()) # Impute with median
    # Formula: |pH - 7| + (TSS / 10)
    df['Pollution_Index'] = abs(df['ph'] - 7.0) + (df[tss_col] / 10.0)
else:
    df['Pollution_Index'] = 0

# --- 3. Data Split (70% Train, 15% Val, 15% Test Logic) ---
# We split 70/30. The 30% Test set is the final "Test". 
# The "Validation" happens internally during Grid Search Cross-Validation (CV).
X = df.drop(columns=[target_col, 'Date', 'Region', 'Station Name', 'River', 'Location', 'Sample ID'], errors='ignore')
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"✅ Data Split: {len(X_train)} Training samples, {len(X_test)} Testing samples.")

# --- 4. Preprocessing (Min-Max Scaling) ---
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Imputation
            ('scaler', MinMaxScaler())                      # Transformation (0-1 Scale)
        ]), numeric_features)
    ],
    remainder='drop'
)

# Process Data for Training
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- 5. Selection (RFE - Recursive Feature Elimination) ---
print("🔍 Running Recursive Feature Elimination (RFE)...")
# We use a standard RF to determine importance for selection
selector = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), step=1)
selector.fit(X_train_processed, y_train)
X_train_selected = selector.transform(X_train_processed)
X_test_selected = selector.transform(X_test_processed)

print(f"   Selected {selector.n_features_} Best Features.")
selected_columns = numeric_features[selector.support_]
print(f"   🌟 Best Features: {list(selected_columns)}")

# --- 6. Tuning (Grid Search with Validation) ---
print("⚙️  Optimizing Hyperparameters (Grid Search)...")

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [10, 20, None],       # Optimize tree depth
    'min_samples_leaf': [1, 2],        # Prevent overfitting
    'sampling_strategy': ['all']       # Handle imbalance
}

base_model = BalancedRandomForestClassifier(random_state=42, bootstrap=False)
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, 
                          cv=StratifiedKFold(n_splits=3), n_jobs=-1, verbose=1, scoring='recall_weighted')

grid_search.fit(X_train_selected, y_train)
best_model = grid_search.best_estimator_
print(f"✅ Best Params: {grid_search.best_params_}")

# --- 7. Evaluation (Recall & Precision Focus) ---
print("📊 Calculating Primary Metrics (Recall & Precision)...")

y_pred = best_model.predict(X_test_selected)
y_proba = best_model.predict_proba(X_test_selected)
classes = best_model.classes_

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))
print("-" * 30)

kappa = cohen_kappa_score(y_test, y_pred)
print(f"✅ COHEN'S KAPPA: {kappa:.4f}")
print("-" * 30 + "\n")

# Save Metrics for App
report_dict = classification_report(y_test, y_pred, output_dict=True)
metrics_data = {"kappa": kappa, "accuracy": report_dict["accuracy"], "report": report_dict}
with open("model_metrics.json", "w") as f:
    json.dump(metrics_data, f)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap='Blues', ax=ax)
plt.title(f"Confusion Matrix (Recall Focus)")
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]
plt.figure(figsize=(10, 6))
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    if i < len(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC of {classes[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# --- 8. Save Final Pipeline ---
# We package Preprocessor -> Selector -> Model
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', selector),
    ('classifier', best_model)
])

joblib.dump(final_pipeline, 'streamsafe_model.pkl')
le = LabelEncoder()
le.fit(y)
joblib.dump(le, 'label_encoder.pkl')

print("💾 Saved Final Production Model.")