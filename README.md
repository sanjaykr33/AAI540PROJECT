# AAI540PROJECT
AAI540-Project-3                                                                                                                               
Project Name: Employee Attrition Prediction 
### Project Scope 
#### Objective  
Develop a machine learning model to predict employee attrition risk (binary 
classification: “Left” vs. “Stayed”) using the synthetic Kaggle Employee Attrition 
Classification Dataset. The model will identify employees at high risk of voluntary 
turnover, enabling HR teams to prioritize proactive retention interventions and reduce 
overall attrition costs. 
#### In Scope 
• End-to-end ML pipeline: data ingestion, exploratory data analysis, preprocessing, 
feature engineering, model training, hyperparameter tuning, evaluation, and 
interpretation. 

• Primary modeling approach: XGBoost (with baseline comparison to Random 
Forest and logistic regression). 

####  Key deliverables: 

o Model achieving ≥85% accuracy and ≥80% recall on holdout data. 

o Ranked list of top attrition drivers with feature importance explanations. 

o Fairness assessment and basic bias checks on sensitive attributes (age, 
gender, marital status). 

o Well-documented Jupyter notebooks and a reproducible pipeline (GitHub). 

• Batch inference design suitable for periodic HR reporting (e.g., monthly risk 
scoring). 

#### Out of Scope 
• Real-time / online prediction serving. 

• Integration with live HRIS systems or real employee data.

• Collection or use of unstructured data (e.g., employee comments, emails). 
                                                                              
• Full production deployment infrastructure (Kubernetes, monitoring dashboards, 
alerting). 

• Large-scale A/B testing of retention interventions based on model outputs. 

• Advanced time-series or survival analysis approaches. 

#### Success Criteria  
A production-viable prototype that reliably flags at-risk employees, provides 
interpretable insights for HR decision-making, and demonstrates responsible AI 
practices (fairness, transparency, documentation) within the constraints of a synthetic 
dataset. 
                                                                                                                             
#### Project Background 
Employee attrition, or the rate at which employees leave a company, poses significant 
challenges for organizations, leading to increased recruitment costs, loss of institutional 
knowledge, and disruptions in team productivity. This project aims to develop a machine 
learning model to predict employee attrition using a synthetic dataset that simulates 
real-world HR scenarios. The model's objective is to identify employees at high risk of 
leaving, enabling proactive interventions by HR teams such as targeted retention 
strategies, career development programs, or workload adjustments. By forecasting 
attrition, companies can reduce turnover rates and foster a more stable workforce. 

This is a supervised binary classification problem in machine learning, where the target 
variable is "Attrition" (categorized as "Left" or "Stayed"). The model will learn patterns 
from historical employee data to classify whether an employee is likely to leave based 
on features like age, job satisfaction, years at the company, and work-life balance. The 
synthetic nature of the dataset ensures ethical training without real personal data, but it 
mirrors common attrition drivers observed in industry studies. 
                                                                                                                             
#### Technical Background 
To evaluate the model, we will use standard classification metrics including accuracy, 
precision, recall, F1-score, and AUC-ROC, with a focus on recall to minimize false 
negatives (missing at-risk employees). The primary data source is the Kaggle Employee 
Attrition Classification Dataset, a synthetic simulation containing approximately 59,598 
rows and 24 columns in the training set, split into train and test files. Data preparation 
involves handling categorical variables through one-hot encoding, scaling numerical 
features, and addressing any imbalances in the target class using techniques like 
SMOTE. 

Data exploration will include exploratory data analysis (EDA) via visualizations such as 
correlation heatmaps, distribution plots, and feature importance rankings from initial 
models. We hypothesize that main features will include job satisfaction, years at 
company, monthly income, and work-life balance, as these often correlate with retention 
in HR analytics. For modeling, we plan to use ensemble methods like Random Forest or 
XGBoost due to their robustness in handling mixed data types and providing 
interpretable feature importances. 
                                           
#### Goals vs Non-Goals 
#### Goals: 
• Develop a predictive model with at least 85% accuracy and 80% recall on the 
test set to reliably identify at-risk employees. 

• Identify key attrition drivers through feature importance analysis to inform HR 
policies. 

• Design a scalable ML pipeline that includes data preprocessing, model training, 
and deployment for potential integration into HR systems. 

• Ensure the system addresses ethical concerns like bias in sensitive features 
(e.g., gender, age). 

• Document the end-to-end process for reproducibility and stakeholder review. 

#### Non-Goals: 
• Implement real-time inference for live employee data streams, focusing instead 
on batch processing. 

• Collect or integrate real employee data from actual companies, sticking to 
synthetic data for this prototype. 

• Optimize for production-scale infrastructure costs, as this is a design exercise. 

• Develop advanced natural language processing for unstructured data like 
employee feedback. 

• Address multi-class predictions beyond binary attrition. 
                                                                                                                          
#### Solution Overview 
The ML system is designed as an end-to-end pipeline for predicting employee attrition, 
starting from data ingestion to model monitoring. Raw data from the Kaggle dataset is 
ingested, preprocessed, and split for training. Feature engineering transforms 
categorical and numerical inputs, followed by model training using XGBoost for its 
efficiency and handling of imbalanced classes. The trained model is deployed as a 
batch inference service, with predictions outputted to a dashboard for HR review. 
Monitoring tracks model drift and performance metrics post-deployment. 

The system architecture includes:  

(1) Data storage in cloud-based S3 buckets or local CSV files;  

(2) Pre-processing via Python scripts with pandas and scikit-learn;  

(3) Feature engineering in a dedicated module;  

(4) Model training/debugging using Jupyter notebooks and MLflow for experiment 
tracking;  

(5) Deployment via Flask API or AWS SageMaker endpoints. We will monitor data drift 
using statistical tests, model performance via weekly retraining triggers, and 
infrastructure health with logging tools like Prometheus. Prior to release, unit tests for 
code, integration tests for the pipeline, and A/B testing on holdout data will be 
conducted. 

Note: System architecture diagram would be included in document, depicting 
components like Data Lake → ETL Pipeline → Feature Store → Model Trainer → 
Serving Layer → Monitoring Dashboard. 
                                                                                                                             
#### Data Sources 

The primary data source is the Kaggle Employee Attrition Classification Dataset, a 
synthetic dataset simulating employee records for attrition analysis. It consists of two 
files: a training set with approximately 59,598 rows and a test set, totaling around 
25,000 rows combined, with 24 columns including features like Employee ID, Age, 
Gender, Years at Company, Job Role, Monthly Income, Work-Life Balance, Job 
Satisfaction, Performance Rating, Number of Promotions, Education Level, Marital 
Status, Number of Companies Worked, Overtime, Distance from Home, Leadership 
Opportunities, Innovation Opportunities, Company Reputation, Employee Recognition, 
Attrition (target), and others such as Time Spent Alone or Social Event Attendance. 
The data volume is modest (under 100,000 rows), making it suitable for local processing 
without big data tools. This dataset was selected because it provides a comprehensive, 
balanced simulation of real HR data without ethical risks from actual employee 
information, allowing focus on model development. It includes a mix of numerical, 
categorical, and ordinal features relevant to attrition prediction, based on common 
industry factors. 

Risks include potential biases in synthetic generation, such as overrepresentation of 
certain demographics (e.g., gender imbalances) or correlations that amplify stereotypes 
(e.g., age and attrition). Sensitive features like gender, age, and marital status could 
introduce fairness issues if not mitigated. No real sensitive data is involved, but we must 
audit for proxy biases. 
