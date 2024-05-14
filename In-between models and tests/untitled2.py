# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:14:35 2024

@author: user
"""

import os
import pandas as pd
from os.path import join
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Change the working directory
os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")

# Load the data
df = pd.read_csv(join('writen data and checkpoints','df_finalv1.csv'))
df = df[df["county_name"] != "massac_illinois"]  

df = df.sort_values(by=['state_name', 'county_name', 'year', 'week'])


train_cutoff_year = 2015
validate_cutoff_year = 2020

train_df = df[df['year'] <= train_cutoff_year]
validate_df = df[(df['year'] > train_cutoff_year) & (df['year'] <= validate_cutoff_year)]
test_df = df[df['year'] > validate_cutoff_year]


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
feature_columns = ['weekly_prcp', 'weekly_srad', 'weekly_tmax', 
                   'weekly_tmin', 'weekly_vp', 'percentage',
                   'percentage_overlap', 'condition_score',
                   'AREA.PLANTED', 'AREA.HARVESTED','PRODUCTION']
# Assuming your feature columns are stored in `feature_columns`
scaler.fit(train_df[feature_columns])  # Fit only on training data

# Transform all datasets
train_df[feature_columns] = scaler.transform(train_df[feature_columns])
validate_df[feature_columns] = scaler.transform(validate_df[feature_columns])
test_df[feature_columns] = scaler.transform(test_df[feature_columns])

weekly_columns = ['weekly_prcp', 'weekly_srad', 'weekly_tmax', 'weekly_tmin', 'weekly_vp', 'percentage', 'percentage_overlap', 'condition_score']
yearly_columns = ['AREA.PLANTED', 'AREA.HARVESTED', 'PRODUCTION', 'Longitude', 'Latitude']

def build_sequences(df, weekly_columns, yearly_columns, max_sequence_length=33):
    # Check if all columns are present in the DataFrame
    required_columns = set(weekly_columns + yearly_columns)
    if not required_columns.issubset(df.columns):
        missing = list(required_columns - set(df.columns))
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    sequences = []
    targets = []
    sequence_years = []
    sequence_counties = []
    
    for _, group in df.groupby(['county_name', 'year']):
        try:
            sequence = group[weekly_columns + yearly_columns].values
            # Ensure sequence length is handled even with missing data
            if len(sequence) < max_sequence_length:
                padding = np.zeros((max_sequence_length - len(sequence), len(weekly_columns) + len(yearly_columns)))
                sequence = np.vstack((sequence, padding))
            elif len(sequence) > max_sequence_length:
                sequence = sequence[:max_sequence_length]
            
            sequences.append(sequence)
            targets.append(group['YIELD'].iloc[0])  # Using .iloc[0] to safely extract the first yield value
            sequence_years.append(group['year'].iloc[0])
            sequence_counties.append(group['county_name'].iloc[0])
        except KeyError as e:
            print(f"Error processing group {group['county_name'].iloc[0]}, {group['year'].iloc[0]}: {e}")
            continue
    
    return np.array(sequences), np.array(targets), np.array(sequence_years), np.array(sequence_counties)

X_train, y_train, _, _ = build_sequences(train_df, weekly_columns, yearly_columns)
X_validate, y_validate, _, _ = build_sequences(validate_df, weekly_columns, yearly_columns)
X_test, y_test, test_years , test_counties = build_sequences(test_df, weekly_columns, yearly_columns)


# Ensure the input shape variables are correctly defined
nx, ny = X_train.shape[1], X_train.shape[2]

# Define and compile the LSTM model
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(nx, ny)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_validate, y_validate), batch_size=64)

# Predict on the test set
y_pred = model.predict(X_test)


# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE: {rmse}')

rmse_normalized = rmse / (y_test.max() - y_test.min())
print("Normalized RMSE:", rmse_normalized)

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')  # Line for perfect predictions
plt.show()

# Plotting residuals
y_pred = y_pred.flatten()
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.show()


def predict_for_county_year(model, X, counties, years, target_county, target_year):
    # Find indices for the specific county and year
    indices = np.where((counties == target_county) & (years == target_year))
    # Extract sequences
    X_target = X[indices]
    # Make predictions
    predictions = model.predict(X_target)
    return predictions

# Example usage
target_county = np.random.choice(test_counties)
target_year = 2021

predicted = predict_for_county_year(model, X_test, test_counties, test_years, target_county, target_year)
true =  y_test[np.where((test_counties == target_county) & ( test_years == target_year))]
