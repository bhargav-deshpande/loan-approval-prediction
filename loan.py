import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("dataset.csv")
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe)
# print(df.isnull().sum())

df["gender"]=df["gender"].fillna(df["gender"].mode()[0])
df["married"]=df["married"].fillna(df["married"].mode()[0])
df["dependents"]=df["dependents"].fillna(df["dependents"].mode()[0])
df["self_employed"]=df["self_employed"].fillna(df["self_employed"].mode()[0])
df["loanamount"]=df["loanamount"].fillna(df["loanamount"].median())
df["loan_amount_term"]=df["loan_amount_term"].fillna(df["loan_amount_term"].median())
df["credit_history"]=df["credit_history"].fillna(df["credit_history"].mode()[0])

x=df.drop(columns=['loan_id','loan_status'])
y=df['loan_status']
y=y.map({"y":1,"n":0})
x["dependents"]=x['dependents'].replace("3+","3")

x_one_encoded=pd.get_dummies(x,
                             columns=['property_area','gender','married','education','self_employed','dependents'],
                             drop_first=True,
                             dtype=int)

x_train, x_test, y_train, y_test = train_test_split(x_one_encoded, y, test_size=0.20, random_state=42,stratify=y)
scaler=StandardScaler()
numeric_col=['applicantincome','loanamount','loan_amount_term','coapplicantincome']
x_train[numeric_col]=scaler.fit_transform(x_train[numeric_col])
x_test[numeric_col]=scaler.transform(x_test[numeric_col])

models={
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ),
    "SVM": SVC(
        probability=True,
        kernel='rbf',
        random_state=42
    )
}

for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(name)
    print("Accuracy: ",accuracy_score(y_test,y_pred))
    print("Recall score: ",recall_score(y_test,y_pred))
    print("F1 Score: ",f1_score(y_test,y_pred),"\n")

base_learners=[
    ('lor',LogisticRegression(max_iter=1000)),
    ('dt',DecisionTreeClassifier(random_state=42)),
    ('rf',RandomForestClassifier(n_estimators=200,max_depth=None,random_state=42)),
    ('svm',SVC(probability=True,kernel='rbf',random_state=42))
]
meta_learner=LogisticRegression(max_iter=1000)

stacking_clf=StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

stacking_clf.fit(x_train,y_train)
y_pred_stack=stacking_clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred_stack)
recall=recall_score(y_test,y_pred_stack)
f1=f1_score(y_test,y_pred_stack)
report=classification_report(y_test,y_pred_stack)
print("Stacking Model")
print("Accuracy :",accuracy)
print("Recall Score: ",recall)
print("F1 Score: ",f1)
print("Classification Report: \n",report)

joblib.dump(stacking_clf,"stacking_model.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(x_one_encoded.columns.tolist(),"columns.pkl")