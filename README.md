# Telco Customer Churn Prediction

## 🚀 Overview
This project uses various machine learning techniques to predict customer churn for a fictional telecommunications company in California. By analyzing data such as customer demographics, service usage, billing information, and additional services, the model identifies key factors driving churn. This insight enables the company to develop targeted strategies to improve customer retention.

## 📖 Description
Telco Customer Churn Prediction is a data-driven project that leverages machine learning to identify and predict customer attrition in the telecommunications industry. This project analyzes a dataset containing 7,043 customer records from a fictional California telco, encompassing details such as demographic information, service usage patterns, billing data, and subscription features. The primary objective is to determine the factors that drive customers to leave the company and develop robust predictive models that flag potential churners early. Comprehensive data preprocessing—including handling missing values, encoding categorical variables, balancing classes with SMOTE, feature selection, and dimensionality reduction via PCA—is combined with various classification algorithms. Ensemble methods such as Random Forest, Gradient Boosting, and XGBoost have demonstrated exceptional performance, providing actionable insights for targeted retention strategies and improved customer satisfaction.

## 🧑‍💻 Features
- **Churn Prediction:** Utilizes factors like contract type, monthly charges, internet service, online security, and more to predict whether a customer will churn.
- **Data Visualization:** Provides visual insights into the relationships between different features and customer churn.
- **Actionable Insights:** Helps decision-makers pinpoint high-risk customers and develop strategies to reduce churn.

## 🔬 Methodology

### Data Preprocessing
- **Handling Missing Values:** Converts problematic columns (e.g., 'Total Charges') to numeric and fills missing values with the column mean.
- **Categorical Variable Encoding:** Encodes categorical features (e.g., 'Gender', 'Senior Citizen', 'Internet Service') using LabelEncoder or one-hot encoding.
- **Feature Engineering:** Drops irrelevant columns such as `CustomerID`, `Count`, `Country`, and `State`, and splits the `Lat Long` column into separate `Latitude` and `Longitude` columns.
- **Data Balancing and Scaling:** Applies SMOTE to address class imbalance and uses StandardScaler to normalize features for improved model performance.

### Model Building and Evaluation
- **Feature Selection & Hyperparameter Tuning:** Uses SelectKBest and GridSearchCV to identify the most influential features and optimize model hyperparameters.
- **Multiple Classifiers:** Evaluates various models including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost using cross-validation and test-set evaluation.
- **PCA Analysis:** Applies Principal Component Analysis (PCA) to reduce dimensionality and assess its effect on model performance.
- **Performance Metrics:** Models are evaluated using accuracy, classification reports, and confusion matrices.

## 📊 Results
Our experiments demonstrate that ensemble methods such as Random Forest, Gradient Boosting, and XGBoost achieve near-perfect performance, with high test accuracy and balanced precision, recall, and F1-scores. These results indicate that our data preprocessing and feature selection strategies successfully capture the key factors driving customer churn.

## Key Insights
- **High Impact Features:** Contract type, monthly charges, and additional services (e.g., online security) are among the strongest predictors of churn.
- **Customer Segmentation:** The model effectively identifies high-risk customer segments, allowing for targeted retention efforts.
- **Robust Performance:** Ensemble methods consistently outperform simpler models, highlighting the complex interactions within the data.

## 📦 Requirements
To run this project, you'll need the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imblearn
- xgboost

## 🏃‍♀️ How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/harshbhati1/Telco-Customer-Churn
2. **Navigate to the project folder:**
   ```bash
   cd TelcoCustomerChurn
3. **Run the notebook: Open the Jupyter Notebook or Google Colab notebook:**
   ```bash
   jupyter notebook TelcoCustomerChurn.ipynb
## 📊 Dataset
Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) from Kaggle.  

## 🙌 Acknowledgments
Special thanks to [Kaggle](https://www.kaggle.com) for providing the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) and to all contributors whose efforts have made this project possible.
