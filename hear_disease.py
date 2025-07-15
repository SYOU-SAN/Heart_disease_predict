import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

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

#print('Logistic Regression number of correct sample: {}'.format(num_correct_samples))
#print('Logistic Regression accuracy: {}'.format(accuracy))

#Random Forsest

X = df[['blood_pressure', 'cholesterol', 'bmi', 'glucose_level', 'gender_Male']]
y = df.heart_disease

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_modle.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_num_correct_samples = accuracy_score(y_test, rf_y_pred, normalize=False)
#print('Random Forest number of correct sample: {}'.format(rf_num_correct_samples))
#print('Random Forest accouracy: {}'.format(rf_accuracy) )

#visualization
#confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(y_test, y_pred_log, 'Logistic Regression')
plot_confusion_matrix(y_test, rf_y_pred, 'Random Forest')

#ROC

def plot_roc_curve(model, X_test_data, y_test_data, title):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_data)[:, 1]
    else:
        y_score = model.decision_function(X_test_data)
        
    fpr, tpr, _ = roc_curve(y_test_data, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} - ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve(log_model, X_test_nor, y_test, 'Logistic Regression')
plot_roc_curve(rf_model, X_test, y_test, 'Random Forest')

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(6, 4))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Random Forest - Feature Importance')
plt.xlabel('Importance')
plt.show()

print(f'Logistic Regression correct samples: {accuracy_score(y_test, y_pred_log, normalize=False)}')
print(f'Logistic Regression accuracy: {log_accuracy:.4f}')
print(f'Random Forest correct samples: {accuracy_score(y_test, rf_y_pred, normalize=False)}')
print(f'Random Forest accuracy: {rf_accuracy:.4f}')
