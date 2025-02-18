# Breast Cancer Prediction with Machine Learning

## 📌 Project Overview
This project aims to predict breast cancer diagnosis using **machine learning classification models**. The **Wisconsin Breast Cancer Dataset** was used to train and evaluate different models. The primary goal was to achieve the highest possible accuracy through **feature engineering, data preprocessing, hyperparameter tuning, and ensemble learning techniques**.

## 📂 Dataset
- **Source:** [Kaggle - Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Features:** 30 numerical attributes representing cell nucleus characteristics extracted from digitalized images
- **Target Variable:**
  - `1` (Malignant - Cancerous)
  - `0` (Benign - Non-cancerous)

## 🚀 Methods Used
### 📊 Data Preprocessing
- **Outlier Detection & Removal**: Local Outlier Factor (LOF) was used to detect and remove noisy data points.
- **Feature Scaling**:
  - Min-Max Normalization
  - Z-Score Normalization
- **Data Splitting**: 80% training, 20% testing
- **Cross-Validation**: 5-Fold Cross Validation

### 🧠 Machine Learning Models
The following classification models were trained and evaluated:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **AdaBoost Classifier**
- **XGBoost Classifier**
- **Voting Classifier (Ensemble Method)**

### 🔧 Hyperparameter Optimization
- `RandomizedSearchCV` was used to find the optimal hyperparameters for each model.
- Best models were ensembled using **Voting Classifier** to improve accuracy and generalization.

## 📈 Performance Evaluation
| Model | Accuracy (No Normalization) | Accuracy (Z-Score) | Accuracy (Min-Max) |
|--------|--------------------|----------------|--------------|
| Logistic Regression | 95.90% | **98.25%** | 97.66% |
| KNN | 95.90% | 95.32% | **96.49%** |
| SVM | 95.32% | 96.49% | 97.66% |
| Decision Tree | 90.05% | 90.05% | 90.05% |
| Random Forest | 96.49% | 96.49% | 96.49% |
| Gradient Boosting | **97.66%** | **97.66%** | **97.66%** |
| AdaBoost | 95.90% | 95.90% | 95.90% |
| XGBoost | **97.66%** | **97.66%** | **97.66%** |
| **Voting Classifier** | 96.49% | **97.66%** | 97.07% |

### 📊 Evaluation Metrics
To assess model performance, the following metrics were used:
- **Accuracy**: Measures the percentage of correctly classified instances.
- **Confusion Matrix**: Provides insight into model classification errors.
- **Classification Report**: Includes precision, recall, and F1-score for each class.
- **ROC Curve & AUC**: Evaluates the model’s ability to distinguish between classes.
- **Feature Correlation Analysis**: Identifies the most significant features contributing to predictions.

----------------------

- **Best performing model:** Logistic Regression with Z-Score Normalization (98.25%)
- **Ensemble learning (Voting Classifier) improved overall accuracy.**
- **Normalization techniques had a significant impact on model performance.**

## 🔬 Key Insights
- **Z-Score Normalization** significantly improved **Logistic Regression** performance.
- **Ensemble Learning (Voting Classifier)** enhanced accuracy and generalization.
- **Gradient Boosting & XGBoost** models performed consistently well across all scenarios.
- **Hyperparameter tuning** with RandomizedSearchCV provided better classification results.


## 📌 Installation & Usage
### 📥 Prerequisites
Ensure you have the following libraries installed:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### 🏃 Running the Code
You can find the complete implementation in the **Jupyter Notebook file** included in this repository.
```bash
jupyter notebook Breast-Cancer-Prediction-ML.ipynb
```
