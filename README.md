# 🧠 Alzheimer’s Disease Progression Prediction
This project leverages clinical and MRI-derived features to predict the presence and severity of Alzheimer’s Disease (AD) using supervised machine learning models. It is based on the OASIS Cross-Sectional MRI dataset, focusing on classifying subjects as either demented or nondemented.

## 📌 Project Objectives
Predict Alzheimer’s disease stages using demographic and MRI-derived features.

Understand how normalized whole brain volume (nWBV), MMSE, and age influence dementia classification.

Build interpretable and performant ML models that can aid in early diagnosis.

## 📊 Dataset Overview
Source: Kaggle - MRI and Alzheimer's Dataset

🧑 Population: 373 individuals (age 60–96)

🧠 Target Variable: CDR (Clinical Dementia Rating), binarized into:

0 — Nondemented

1 — Demented

Key Features:

Demographic: Age, Gender, Education, SES, Handedness

Cognitive: MMSE, CDR

MRI: eTIV, nWBV, ASF

## ⚙️ Preprocessing Steps
Missing values imputed with median (for SES and MMSE)

One-hot encoding for categorical variables (Gender, Handedness)

Target binarization: CDR mapped to {0, 1}

Stratified 80/20 train-test split for balanced evaluation

## 🧠 Modeling
🔹 Baseline Model: Decision Tree Classifier
Interpretable and fast

Tuned via GridSearchCV

ROC-AUC: ~0.896

🔸 Final Model: XGBoost Classifier
High accuracy and robustness

Handles missing values internally

Tuned Hyperparameters:

n_estimators = 200

max_depth = 3

learning_rate = 0.1

ROC-AUC: ~0.951

## 📈 Evaluation Metrics
Accuracy

Precision / Recall / F1-score

Confusion Matrix

Stratified 5-Fold Cross-Validation

## 🔍 Explainability with SHAP
Used SHAP (SHapley Additive exPlanations) to interpret feature impact

Key insights:

Higher nWBV reduces dementia risk

Higher Age increases risk

MMSE shows variable impact depending on score

## 📊 Visualizations
Correlation heatmaps

Histograms and boxplots (e.g., nWBV by class)

SHAP summary and dependence plots

Scatter plots of Age vs. Brain Volume

# 🧪 How to Run
## 🔧 Requirements
```bash
pip install pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, shap
```
## ▶️ Run the notebook
```bash
jupyter notebook alzheimer.ipynb
```
## 📌 Key Findings
MMSE, nWBV, ASF, and Age are top predictors.

The model provides solid accuracy with explainability.

Potential to extend this work into a clinical decision support system.

## ⚠️ Limitations
Small dataset (~373 samples)

Class imbalance: fewer demented cases

No temporal component (cross-sectional only)
