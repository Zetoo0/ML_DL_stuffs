# -*- coding: utf-8 -*-
"""
Modulok Importálása
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

"""Adatok betöltése és felosztása"""

melb_data = pd.read_csv('https://raw.githubusercontent.com/karsarobert/Machine_Learning_2024/main/melb_data.csv')
melb_x_data = melb_data.drop(["Price"],axis=1)
x = melb_x_data.select_dtypes(exclude=['object'])
y = melb_data.Price

x

"""Adatok szétvágása 80-20% arányban"""

x_train,x_valid,y_train,y_valid = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)

#Select numerical columns
numerical_cols = [col for col in x_train if x_train[col].dtype in ['int64', 'float64']]

#Select categorical columns
object_cols = [col for col in x_train.columns if x_train[col].nunique() < 10 and x_train[col].dtype == "object"]

print(f"Numerical columns: {numerical_cols}")
print(f"Categorical columns: {object_cols}")

#Merge the  numerical and categorical columns
my_columns = numerical_cols + object_cols
x_train_ = x_train[my_columns].copy()
x_valid_ = x_valid[my_columns].copy()

print(f"x train dataset length: {len(x_train_)}, x validation dataset length: {len}")

x_train_

"""Imputálás és előfeldolgozás"""

#Imputation for numerical datas
num_imputer = SimpleImputer(strategy="mean")#pótlás mediánnal

#Imputation and One Hot Encoding for categorical datas
cat_imputer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

#numerical and categorical columns preprocessing for the pipeline
preprocess = ColumnTransformer(
    transformers=[
        ('numerical', num_imputer, numerical_cols),
        ('categorical', cat_imputer, object_cols)
    ]
)

"""Model elkészítése és tanítása"""

#Creating the Linear Regression model
model = LinearRegression()

#Create the pipeline
pipeline = Pipeline(steps=[('preporocessor',preprocess),
                           ('model', model)])

pipeline.fit(x_train_,y_train)

#training prediction
train_predict = pipeline.predict(x_train_)

#validation set prediction
valid_predict = pipeline.predict(x_valid)

"""Kiértékelés

"""

mae = mean_absolute_error(y_valid,valid_predict)
r2 = r2_score(y_valid, valid_predict)

print(f"MEA: {mae}")
print(f"R^2: {r2}")