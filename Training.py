import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.metrics import classification_report

df = pd.read_csv('diabetes_clean.csv')

X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome', axis=1), df['Outcome'], test_size=0.2, random_state=42)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification report = \n", classification_report(y_test, y_pred))