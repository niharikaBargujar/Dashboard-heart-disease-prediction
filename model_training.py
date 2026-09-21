import os
import shutil
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Create directories
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load dataset
heart_src = os.path.join(os.path.dirname(base_dir), 'heart.csv')
heart_dest = os.path.join(data_dir, 'heart.csv')
if os.path.exists(heart_src):
    shutil.copy(heart_src, heart_dest)

df = pd.read_csv(heart_dest)
print(f"Loaded raw heart dataset: {df.shape}")
df_clean = df.drop_duplicates()
print(f"Cleaned dataset: {df_clean.shape}")
df_clean.to_csv(heart_dest, index=False)

# Features & Target
X = df_clean.drop('target', axis=1)
y = df_clean['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Random Forest (Tuned as in Colab: n_estimators=200, max_depth=5)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# 2. Logistic Regression (Tuned as in Colab: C=1, solver='liblinear')
log_model = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# 3. Support Vector Classifier (Tuned as in Colab: C=10, kernel='linear', probability=True)
svm_model = SVC(C=10.0, kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

# Save models
joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_model.joblib'))
joblib.dump(log_model, os.path.join(models_dir, 'logistic_regression_model.joblib'))
joblib.dump(svm_model, os.path.join(models_dir, 'svm_model.joblib'))

# Calculate metrics
def compute_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'report': classification_report(y_true, y_pred, output_dict=True)
    }

metrics = {
    'features': list(X.columns),
    'dataset_stats': {
        'total_samples': int(len(df_clean)),
        'features_count': int(X.shape[1]),
        'target_distribution': df_clean['target'].value_counts().to_dict(),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    },
    'Random Forest': {
        **compute_metrics(y_test, y_pred_rf),
        'feature_importances': dict(zip(X.columns, [float(x) for x in rf_model.feature_importances_]))
    },
    'Logistic Regression': {
        **compute_metrics(y_test, y_pred_log),
        'coefficients': dict(zip(X.columns, [float(x) for x in log_model.coef_[0]]))
    },
    'Support Vector Machine': {
        **compute_metrics(y_test, y_pred_svm)
    }
}

with open(os.path.join(models_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print("Saved all models and metrics!")
print(f"Random Forest Accuracy: {metrics['Random Forest']['accuracy']:.4f}")
print(f"Logistic Regression Accuracy: {metrics['Logistic Regression']['accuracy']:.4f}")
print(f"SVM Accuracy: {metrics['Support Vector Machine']['accuracy']:.4f}")

# Create sample batch test CSV for users to try batch prediction
sample_batch = X_test.copy().head(20)
sample_batch.to_csv(os.path.join(data_dir, 'sample_batch_test.csv'), index=False)
print("Created data/sample_batch_test.csv with 20 test records.")
