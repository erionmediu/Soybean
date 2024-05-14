# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:11:46 2024

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
from os.path import join
import warnings

warnings.filterwarnings('ignore')

# Set the working directory
os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")

# Load the data
df = pd.read_csv(join('writen data and checkpoints', 'df_finalv3_daily.csv'))
df = df[df["county_name"] != "massac_illinois"]  # Filtering example
df = df.drop('state_name', axis=1)

# Convert 'week_start' to a proper date and sort the DataFrame
df['date'] = pd.to_datetime(df['year'].astype(str), format='%Y') + pd.to_timedelta(df['day'] - 2, unit='d')
df.sort_values(by=['county_name', 'date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Preparing features and target
droplist = ['county_name', 'stage', 'overlapping', 'condition_class', 'date', 'YIELD', 'year', 'day']
features = df.drop(droplist + ['YIELD'], axis=1).columns

# Set up the scaler
scaler = StandardScaler()

# Iterative learning
counties = df['county_name'].unique()
years = df['year'].unique()
final_results = pd.DataFrame()

for county in counties:
    county_data = df[df['county_name'] == county]
    for year in years:
        annual_data = county_data[county_data['year'] == year]
        if annual_data.empty:
            continue

        # Split the data - here assuming you're predicting for the next available year
        train_data = annual_data[annual_data['year'] < year]
        test_data = annual_data[annual_data['year'] == year]

        if train_data.empty or test_data.empty:
            continue

        X_train = train_data[features]
        y_train = train_data['YIELD']
        X_test = test_data[features]
        y_test = test_data['YIELD']

        # Scaling
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Store results
        results = test_data[['county_name', 'year']].copy()
        results['Predicted_YIELD'] = y_pred
        results['Actual_YIELD'] = y_test.reset_index(drop=True)

        final_results = pd.concat([final_results, results])

# Display the final DataFrame
print(final_results)

# Calculate the Mean Squared Error over all predictions
mse = mean_squared_error(final_results['Actual_YIELD'], final_results['Predicted_YIELD'])
print(f'Mean Squared Error: {mse}')