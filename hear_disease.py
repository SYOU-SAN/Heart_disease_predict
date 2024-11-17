import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/user/Desktop/test_01/Heart_Disease_and_Hospitals.csv')
df = pd.get_dummies(df, columns=['gender'], dtype=int, drop_first=True)

df.head()

X = df[['blood_pressure', 'cholesterol', 'bmi', 'glucose_level', 'gender_Male']]
#y = df['heart_disease']
#df = df.drop(['full_name', 'country', 'state', 'first_name', 'last_name', 'hospital', 'treatment_date', 'treatment'], axis=1)

#X = df.drop('heart_disease', axis=1)
y = df.heart_disease

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)

#Logistic Regression

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test_nor = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test_nor)
accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)

print('Logistic Regression number of correct sample: {}'.format(num_correct_samples))
print('Logistic Regression accuracy: {}'.format(accuracy))

#Random Forsest

X = df[['blood_pressure', 'cholesterol', 'bmi', 'glucose_level', 'gender_Male']]
y = df.heart_disease

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)

rf_modle = RandomForestClassifier(n_estimators=100, random_state=42)
rf_modle.fit(X_train, y_train)
rf_y_pred = rf_modle.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_num_correct_samples = accuracy_score(y_test, rf_y_pred, normalize=False)
print('Random Forest number of correct sample: {}'.format(rf_num_correct_samples))
print('Random Forest accouracy: {}'.format(rf_accuracy) )