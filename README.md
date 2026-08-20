# Loan Approval Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a loan application is likely to be approved or rejected using machine learning classification algorithms.

The project compares multiple classification models and uses a **Stacking Ensemble** to combine their predictions. The final trained model is deployed as an interactive **Streamlit web application**.

## 🚀 Features

* Data preprocessing and missing-value handling
* One-Hot Encoding for categorical variables
* Feature scaling using StandardScaler
* Multiple classification algorithms
* Stacking Ensemble Learning
* Model evaluation using Accuracy, Recall and F1 Score
* Interactive Streamlit application

## 🤖 Machine Learning Models

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Stacking Classifier

### Stacking Ensemble

The Stacking Classifier uses:

**Base Learners**

* Logistic Regression
* Decision Tree
* Random Forest
* SVM

**Meta Learner**

* Logistic Regression

## 📊 Model Evaluation

The models were evaluated using:

* Accuracy
* Recall
* F1 Score
* Classification Report

| Model               |   Accuracy |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: |
| Logistic Regression |   0.8617   |   0.9882   |   0.9081   |
| Decision Tree       |   0.7317   |   0.7764   |   0.8      |
| Random Forest       |   0.8373   |   0.9411   |   0.8888   |
| SVM                 |   0.8455   |   0.9882   |   0.8983   |
| Stacking Classifier |   0.8617   |   0.9882   |   0.9081   |

## 🖥️ Streamlit Application

The trained Stacking Ensemble model is deployed using Streamlit.

Users can enter loan application details such as:

* Gender
* Marital Status
* Dependents
* Education
* Self Employment
* Applicant Income
* Co-Applicant Income
* Loan Amount
* Loan Term
* Credit History
* Property Area

The application then predicts whether the loan is:

**Loan Approved** or **Loan Rejected**

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit

## 📁 Project Structure

```text
loan-approval-prediction/
│
├── loan.py
├── app_deploy.py
├── dataset.csv
├── stacking_model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/bhargav-deshpande/loan-approval-prediction.git
```

Move into the project directory:

```bash
cd loan-approval-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Machine Learning Model

```bash
python loan.py
```

## 🌐 Run the Streamlit Application

```bash
python -m streamlit run app_deploy.py
```

The application will open in your browser.

## 🔮 Future Improvements

* Hyperparameter tuning
* Cross-validation and more extensive model comparison
* Improved preprocessing using Pipeline and ColumnTransformer
* Model explainability
* Deployment with a public live URL

## 👨‍💻 Author

Bhargav Deshpande