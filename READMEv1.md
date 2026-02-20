# Employee Attrition Prediction using AWS SageMaker

## Project Overview
This project aims to predict employee attrition (whether an employee will leave the company) using supervised machine learning. The solution is built using AWS SageMaker and demonstrates a complete end-to-end machine learning workflow.

The project covers data ingestion, exploratory data analysis (EDA), data cleaning, feature engineering, model training, evaluation, and deployment using managed AWS services.

---

## Problem Statement
Employee attrition can significantly impact organizational performance and costs.  
The objective of this project is to build a classification model that predicts whether an employee is likely to leave the company based on demographic, job-related, and organizational features.

---

## Dataset
- Source: Kaggle – Employee Attrition Dataset
- Target Variable: `Attrition`
  - `0` → Stayed
  - `1` → Left
- Features include:
  - Age, Monthly Income, Years at Company
  - Job Role, Job Level, Company Size
  - Work-Life Balance, Job Satisfaction
  - Remote Work, Leadership Opportunities, Innovation Opportunities
  - Company Reputation, Employee Recognition

---

## Project Structure

employee-attrition-ml/     
│    
├── notebooks/    
│ ├── 01_data_ingestion_eda.ipynb    
│ ├── 02_data_cleaning_feature_engineering.ipynb    
│ ├── 03_model_training_sagemaker.ipynb    
│ └── 04_evaluation_and_deployment.ipynb    
│    
├── README.md    
├── requirements.txt    
└── .gitignore    
---

## Workflow Summary

### 1. Data Ingestion & EDA
- Loaded raw training and test datasets
- Performed exploratory data analysis
- Checked data distributions and target balance

### 2. Data Cleaning & Feature Engineering
- Handled categorical variables using encoding
- Scaled numerical features
- Encoded target labels
- Split data into train, validation, and test sets
- Stored processed datasets in Amazon S3

### 3. Model Training (AWS SageMaker)
- Used XGBoost classifier
- Trained model using SageMaker managed training jobs
- Stored trained model artifacts in S3 / Model Registry

### 4. Evaluation & Deployment
- Deployed model to a real-time SageMaker endpoint
- Performed inference on test data
- Evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC
- Deleted endpoint after evaluation to avoid costs

---

## Model Performance (Test Set)

- Accuracy: ~65%
- Precision: ~64%
- Recall: ~61%
- F1 Score: ~63%
- ROC-AUC: ~65%

These results represent a strong baseline model for employee attrition prediction.

---

## Tools & Technologies
- Python
- Pandas, NumPy, Scikit-learn
- XGBoost
- AWS SageMaker (Training, Deployment, Endpoints)
- Amazon S3
- Git & GitHub

---

## Notes
- The project demonstrates proper ML lifecycle management and cloud resource handling.
- Endpoints and compute resources were cleaned up after use.
- The notebooks are intended as execution records and not for repeated deployment without modification.

---

## Author
Ahmed Ali, Sanjay Kumar

