import pandas as pd
import numpy as np

from sklearn.metrics import explained_variance_score as evs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


# Data Import
df = pd.read_csv('medical_expenses.csv')


# EDA & Preprocessing
print(df.info())

print(df)

df['charges'] = round(df['charges'])
df['bmi'] = round(df['bmi'], 1)

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    df[col] = df[col].astype('float')


# Train Test Split
features = df.drop('charges', axis= 1)
target = df['charges']

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size= 0.25, shuffle= True, random_state= 42)


# Model Training
models = [LinearRegression(), LogisticRegression(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for m in models:
    print(m)

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy : {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy : {evs(Y_test, pred_test)}\n')