# CardioPredict AI: Heart Disease Risk Stratification Dashboard

A full-featured, clinical-grade interactive web dashboard built with **Streamlit**, **Scikit-Learn**, and **Plotly**, replicating the architecture of modern predictive dashboards (such as the Student Dropout Risk Predictor).

---

## 🚀 Features

1. **🩺 Individual Patient Diagnosis**:
   - Interactive clinical parameter inputs with sliders, radios, and dropdowns.
   - Dynamic real-time risk gauge meter (0% to 100% disease probability).
   - Clinical alert badges (Low Risk vs High Risk) and actionable clinical recommendations.
   - Key vital factor flags (Resting BP, Serum Cholesterol, Max Heart Rate, ST Depression).

2. **📁 Batch Patient Screening**:
   - Drag-and-drop `.csv` file upload for screening multiple patient records at once.
   - Sample CSV template download button (`sample_patients_batch.csv`).
   - Automated risk classification, triage priority tagging (`Urgent`, `Moderate`, `Normal`), and cohort statistics.
   - Cohort pie chart and probability distribution histogram.
   - Export analyzed batch predictions to CSV.

3. **📊 Model Benchmarks & Explainable AI (XAI)**:
   - Performance comparison matrix (Accuracy, Precision, Recall, F1) across **Random Forest (83.6%)**, **Logistic Regression (78.7%)**, and **Support Vector Machine (78.7%)**.
   - Interactive Confusion Matrix heatmaps.
   - Feature Importance charts (Gini importance & regression weights).
   - Full per-class classification reports.

4. **📈 Exploratory Data Analysis (EDA)**:
   - Target balance chart.
   - Chest Pain type vs Disease incidence.
   - Age vs Maximum Heart Rate (`thalach`) regression scatter plot.
   - Serum Cholesterol box plots across risk cohorts.
   - Pearson correlation matrix heatmap across all 13 clinical attributes.

5. **📖 Clinical Dictionary & Methodology**:
   - Medical definitions, reference ranges, and clinical significance for every variable.
   - Pipeline documentation (deduplication, feature splits, hyperparameter tuning).

---

## 🛠️ Quick Start

### 1. Installation
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
Double-click `run_dashboard.bat` on Windows, or run:
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.
