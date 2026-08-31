# 🏦 Bank Customer Churn Prediction

## 📋 Executive Summary
Customer retention is a critical metric in the banking industry, where the cost of acquiring a new customer is significantly higher than retaining an existing one. This project develops a machine learning pipeline to predict customer attrition with a focus on **Recall**. By identifying at-risk customers early, the bank can deploy targeted interventions to reduce revenue loss.

The final **XGBoost** model achieved a 80% Accuracy rate, and a **79% Recall rate** allowing the bank to proactively target approximately  4 out of 5 potential churners.

## 💻 Technologies Used
* **Python:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest, KNN Imputer)
* **Explainability:** SHAP

## 📋 Prerequisites
To run this project, you will need:
* **Python 3.12.4** (or higher)
* **Jupyter Notebook** (installed via Anaconda or pip)

## 💼 Business Context & Strategy
* **The Problem:** Customer churn leads to lost revenue and wasted acquisition spend.
* **The Goal:** Build a model to segment customers based on their probability of churning.
* **The Strategy (Recall-First):** * **False Negatives (Missed Churn):** High Cost. We lose the customer lifetime value.
    * **False Positives (Wrongly Flagged):** Low Cost. We send a discount to a happy customer, slightly reducing margin but increasing loyalty.
    * **Decision:** We optimize hyperparameters for **Recall** to minimize False Negatives, prioritizing the identification of at-risk clients over perfect precision.

## 📊 Data Overview
The dataset contains **10,000 records** of bank customers, including demographic, financial, and engagement attributes.

| Feature Category | Attributes |
| :--- | :--- |
| **Demographics** | `Age`, `Gender`, `Geography` |
| **Financials** | `CreditScore`, `Balance`, `EstimatedSalary`, `NumOfProducts`, `HasCrCard` |
| **Engagement** | `Tenure`, `IsActiveMember` |
| **Target** | `Exited` (0 = Retained, 1 = Churned) |


## 🛠️ Methodology
1.  **Data Preprocessing:**
    * **Data Integrity:** Handled duplicates and formatted inconsistent labels.
    * **Imputation:** Used **K-Nearest Neighbors (KNN)** to impute missing `Age` values and erroneous negative `EstimatedSalary` entries, preserving data variance better than mean imputation.
    * **Scaling:** Applied StandardScaler to normalize numerical features for logistic regression.
2.  **Exploratory Data Analysis (EDA):**
    * Identified significant class imbalance (20% Churn vs. 80% Retained).
    * Noted overlap in feature distributions, suggesting non-linear models would outperform linear ones.
3. Model Development
    * **Optimization Attempts:** Utilized `RandomizedSearchCV` to tune multiple hyperparameters (including `class_weight='balanced'`).
    * **Baseline (Logistic Regression):** Initial model showed **71% Recall** but failed on Precision (**38%**)
    * **Strategic Pivot (Random Forest, XGBoost):** To achieve better overall performance, I transitioned to a **Random Forest classifier**, and then **XGBoost** model. After hyperparameter tuning, XGBoost final model achieved the best balance: **Accuracy 80%** and **Recall 79%** on the test set.

## 📈 Key Results
| Model | Recall Score | Accuracy | Observations |
| :--- | :--- | :--- |:--- |
| **Logistic Regression (Tuned)** | 71% | 72% |Establishes a solid baseline with 71% recall. |
| **Random Forest (Tuned)** | **73%** | 80% |Offers a better balance of accuracy and recall, capturing non-linear relationships. |
| **XGboost (Tuned)** | **79%** | 80% |**Best Performer.** Stands out as the ultimate best performer with 79% recall, offering the optimal precision-recall trade-off for complex data. |

## 🔍 Insights & Strategic Recommendations
Using **SHAP (SHapley Additive exPlanations)** values, we identified the key drivers of churn and formulated the following strategies:

1.  **The Senior Retention Program:** * *Insight:* Age is the dominant driver; customers over 40 are high-risk.
    * *Action:* Audit product usability and fee structures for this demographic.
2.  **Germany Regional Audit:**
    * *Insight:* German customers churn at a disproportionately high rate.
    * *Action:* Investigate local competitor offers or customer service issues specific to the region.
3.  **Product Bundle Review:**
    * *Insight:* Customers with 3+ products are highly unstable.
    * *Action:* Investigate if "over-bundling" leads to hidden fees or complexity that frustrates high-value clients.

---
*Author: Karolina Marek*
---
## ℹ️ Data Source & License

* **Dataset:** Bank Customer Churn
* **Source:** Kaggle (provided by Maven Analytics)
* **License:** Public Domain
