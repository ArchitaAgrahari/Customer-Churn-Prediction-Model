# 📉 Customer Churn Prediction

A Machine Learning project to predict whether a customer will leave a bank service based on demographic and account-related features.

## 🧠 Problem Statement

Customer churn is a major challenge in the banking industry. This project aims to build a predictive model to identify customers who are likely to exit, allowing the bank to take proactive steps to retain them.

---

## 📊 Dataset

- **Source**: [Kaggle - Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- **Target Variable**: `Exited`  
  - `1`: Customer left  
  - `0`: Customer stayed

### 📁 Features:

- **Categorical**: `Geography`, `Gender`
- **Numerical**: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`, etc.
- **Identifiers** (Dropped during preprocessing): `RowNumber`, `CustomerId`, `Surname`

---

## 🔍 Step-by-Step Approach

### ✅ Step 1: Understanding the Dataset
- Explored types of features (categorical, numerical, ID).
- Identified target variable.

### ✅ Step 2: Exploratory Data Analysis (EDA)
- Visualized churn distribution (found class imbalance).
- Plotted histograms and boxplots of important features.
- Analyzed correlation with churn.

### ✅ Step 3: Preprocessing
- Dropped identifier columns.
- Label encoded `Gender` and one-hot encoded `Geography`.
- Scaled numerical features using `StandardScaler`.

### ✅ Step 4: Model Building
- Models Implemented:
  - Random Forest Classifier
  - XGBoost Classifier
- Evaluated with:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📈 Model Evaluation Results

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Random Forest       | 0.86     | 0.75      | 0.62   | 0.68     | 0.84    |
| XGBoost             | 0.88     | 0.78      | 0.64   | 0.70     | 0.87    |

---

## 💡 Key Insights

- **Age** is the most important feature — older customers churn more often.
- **Geography** matters — customers from **Germany** showed higher churn.
- **Tenure** and **Balance** have moderate influence.
- **Gender** has less influence on churn probability.
- XGBoost outperforms other models across most metrics.

---


