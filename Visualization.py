import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('diabetes.csv')
feature_distribution = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
all_feature = [col for col in df.loc[:,:'Age'].columns]

def data_distribution(feature, data) :
    n_cols = 2
    n_rows = (len(feature) + 1) // n_cols

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12,12))

    for i, column in enumerate(feature):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        sns.histplot(data=data, x=column, hue='Outcome', kde=True, ax=ax)
        ax.set_title(f"{column} distribution")
    plt.tight_layout()   
    return plt.savefig('Visualization of data\distribution_of_data.jpg', dpi=300)

def check_imbalance(data):
    plt.figure(figsize=(12,12))
    sns.countplot(data=data, x=data['Outcome'])
    plt.title("Output Size")
    plt.tight_layout()

    return plt.savefig('Visualization of data\data balance.jpg', dpi=300)

def heatmap(data):
    plt.figure(figsize=(12,12))

    corr = data.corr().round(2)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, cbar=True, mask=mask)
    plt.title("Heatmap")

    return plt.savefig('Visualization of data\heatmap.jpg', dpi=300)

def boxplot(data, feature):
    plt.figure(figsize=(10,30))

    n_cols = 2
    n_rows = (len(feature) + 1) // n_cols
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,20))
    for i, column in enumerate(feature):
        col = i % n_cols
        row = i // n_cols
        ax = axs[row, col]
        sns.boxplot(data=data, x=column, ax=ax)
        ax.set_title(f"{column} Boxplot")
    plt.tight_layout()
    return plt.savefig('Visualization of data\/boxplot.jpg', dpi=300)

data_distribution(feature_distribution, df)
check_imbalance(df)
heatmap(df)
boxplot(df, all_feature)