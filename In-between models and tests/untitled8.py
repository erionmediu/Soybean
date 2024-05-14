# -*- coding: utf-8 -*-
"""
Created on Mon May 13 00:15:00 2024

@author: user

REASON THE HASH FEATURE MODEL
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from os.path import join

# Change the working directory
os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")

# Load the data
df = pd.read_csv(join('writen data and checkpoints', 'df_finalv3_daily.csv'))
df = df[df["county_name"] != "massac_illinois"]  
df = df[df["county_name"] == "adams_illinois"]
df = df.drop('state_name', axis=1)
OBJ = ['county_name','YIELD', 'year', 'tmax']
df = df[OBJ]

# Prepare data by extending weekly and yearly data
df['week_start'] = df['day'] - df['day'].mod(7)
df['date'] = pd.to_datetime(df['year'].astype(str), format='%Y') + pd.to_timedelta(df['week_start'] - 1, unit='d')
df.sort_values(by='date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Drop unnecessary columns
df.drop(['year', 'week_start', 'county_name', 'stage', 'overlapping', 'condition_class'], axis=1, inplace=True)

# Feature hashing
hasher = FeatureHasher(n_features=1, input_type='string')

# Hash the 'tmax' column
def hash_features(df):
    df['tmax_str'] = df['tmax'].astype(str)
    hashed_features = hasher.transform(df[['tmax_str']].to_dict(orient='records'))
    hashed_df = pd.DataFrame(hashed_features.toarray())
    return hashed_df

# Split the data into training and test sets
X = df.drop(['YIELD'], axis=1)  # Assuming 'date' is still in your dataframe
y = df['YIELD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform pipeline
transform_pipeline = make_pipeline(
    FunctionTransformer(hash_features, validate=False),
    StandardScaler()
)

X_train_transformed = transform_pipeline.fit_transform(X_train)
X_test_transformed = transform_pipeline.transform(X_test)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Predict on the test set
y_pred = model.predict(X_test_transformed)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plotting
sort_idx = np.argsort(y_test)
y_test_sorted = y_test.iloc[sort_idx]
y_pred_sorted = y_pred[sort_idx]

plt.figure(figsize=(10, 6))
plt.plot(y_test_sorted.reset_index(drop=True), label='Actual Yield')
plt.plot(y_pred_sorted, label='Predicted Yield', linestyle='--')
plt.title('True vs. Predicted Yields')
plt.xlabel('Sorted Test Samples')
plt.ylabel('Yield')
plt.legend()
plt.show()
