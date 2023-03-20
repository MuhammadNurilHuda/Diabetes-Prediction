import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_csv('diabetes.csv')

zero_features = ['Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI']
feature_names = [cname for cname in df.loc[:,:'Age'].columns]

def remove_outlier(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1

    data = data[~((data[feature] < (Q1 - 1.5*IQR)) | (data[feature]> (Q3 + 1.5*IQR))).any(axis=1)]
    
    return data.to_csv('diabetes_outlier_removed.csv', index=False)



def remove_zero(data, feature):
    data[feature].replace(0, np.nan, inplace=True)
    data[feature].dropna(inplace=True)
    return data.to_csv('diabetes_clean.csv', index=False)


remove_outlier(df, feature_names)
df_outlier_removed = pd.read_csv('diabetes_outlier_removed.csv')
remove_zero(df_outlier_removed, zero_features)
