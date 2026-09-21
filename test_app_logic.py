import os
import json
import joblib
import pandas as pd

base = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base, 'models')
data_dir = os.path.join(base, 'data')

# 1. Test model loading
rf = joblib.load(os.path.join(models_dir, 'random_forest_model.joblib'))
lr = joblib.load(os.path.join(models_dir, 'logistic_regression_model.joblib'))
svm = joblib.load(os.path.join(models_dir, 'svm_model.joblib'))
print("Successfully loaded all 3 models.")

# 2. Test metrics.json
with open(os.path.join(models_dir, 'metrics.json')) as f:
    m = json.load(f)
assert 'Random Forest' in m
assert 'Logistic Regression' in m
assert 'Support Vector Machine' in m
print("Metrics JSON verified.")

# 3. Test single patient inference
sample_patient = pd.DataFrame([{
    'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125, 'chol': 212,
    'fbs': 0, 'restecg': 1, 'thalach': 168, 'exang': 0, 'oldpeak': 1.0,
    'slope': 2, 'ca': 2, 'thal': 3
}])

for name, model in [('Random Forest', rf), ('Logistic Regression', lr), ('SVM', svm)]:
    pred = model.predict(sample_patient)[0]
    prob = model.predict_proba(sample_patient)[0]
    print(f"{name}: Prediction = {pred}, Probs = {prob}")

# 4. Test batch test CSV
batch_df = pd.read_csv(os.path.join(data_dir, 'sample_batch_test.csv'))
batch_preds = rf.predict(batch_df)
print(f"Batch prediction on {len(batch_preds)} records successful!")
print("ALL TESTS PASSED!")
