!pip install eli5
import os
os.makedirs('/content/data', exist_ok=True)
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!mkdir -p /content/data
!kaggle datasets download -d taweilo/loan-approval-classification-data -p /content/data
!unzip /content/data/loan-approval-classification-data.zip -d /content/data
import pandas as pd

df = pd.read_csv('/content/data/loan_data.csv')

df.head()
from sklearn.preprocessing import LabelEncoder

perm_imp = df.copy()

for col in ['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file']:
    le = LabelEncoder()
    perm_imp[col] = le.fit_transform(perm_imp[col])

perm_imp.head()

X = perm_imp.drop(columns=['loan_status'])
y = perm_imp['loan_status']
perm_imp.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

features = ['previous_loan_defaults_on_file','loan_int_rate','loan_percent_income','person_home_ownership','person_income']
X = perm_imp[features]

X.head()

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=1)

# Fit model
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model.fit(train_X, train_y)

print(model.predict(val_X.head()))

from sklearn.metrics import accuracy_score, classification_report

preds = model.predict(val_X)

print("Accuracy:", accuracy_score(val_y, preds))
print(classification_report(val_y, preds))

from sklearn.metrics import confusion_matrix, roc_auc_score

preds = model.predict(val_X)
probs = model.predict_proba(val_X)[:,1]

print(confusion_matrix(val_y, preds))
print("ROC-AUC:", roc_auc_score(val_y, probs))
