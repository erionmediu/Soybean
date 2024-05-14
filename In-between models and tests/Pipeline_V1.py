# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:06:18 2024

@author: Erion Mediu
"""

import os 
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from os.path import join 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")


df = pd.read_csv(join('writen data and checkpoints','df_finalv1.csv'))
df = df[df["county_name"]!= "massac_illinois"] # MASACC ILLINOIS TMAX DATA IN 1980 
# MUST BE RE DOWNLOADED THIS MEANS --> ORCHESTRATE DOWNLOAD AND PROCCESS NC FILES 
# test = df[df['weekly_tmax'].isna()== True]

df.info()
df.describe()
df.columns

# Filter for weeks 14 to 51
df_filtered = df[(df['week'] >= 1) & (df['week'] <= 52)]

# Sort the data by county and year, ensuring that weekly data is in the correct order
df_sorted = df_filtered.sort_values(by=['state_name', 'county_name', 'year', 'week'])


# =============================================================================
# 
# def build_sequences(df, weekly_columns, yearly_columns, max_sequence_length=38):
#     sequences = []
#     targets = []
#     
#     # Ensure data is sorted correctly
#     df_sorted = df.sort_values(by=['state_name', 'county_name', 'year', 'week'])
# 
#     # Iterate over each group (unique combination of county and year)
#     for _, group in df_sorted.groupby(['county_name', 'year']):
#         # Get weekly data
#         weekly_data = group[weekly_columns].values
#         
#         # Get yearly data replicated for each week in the sequence
#         yearly_data = np.repeat(group[yearly_columns].iloc[0:1].values, len(weekly_data), axis=0)
# 
#         # Combine weekly and yearly data
#         sequence = np.hstack((weekly_data, yearly_data))
#         
#         # Pad sequence if it is shorter than the max length
#         if len(sequence) < max_sequence_length:
#             padding = np.zeros((max_sequence_length - len(sequence), sequence.shape[1]))
#             sequence = np.vstack((sequence, padding))
#         elif len(sequence) > max_sequence_length:
#             sequence = sequence[:max_sequence_length]  # Truncate sequence if needed
#         
#         # Target yield value
#         target = group['YIELD'].values[0]  # Assume YIELD is consistent within the group
#         
#         sequences.append(sequence)
#         targets.append(target)
#     
#     return np.array(sequences), np.array(targets)
# =============================================================================

# Define weekly and yearly columns
weekly_columns = ['weekly_prcp', 'weekly_srad', 'weekly_tmax', 'weekly_tmin', 
                  'weekly_vp', 'percentage' ,'percentage_overlap',
                  'condition_score'] # 'condition_class', 'overlapping'
yearly_columns = ['AREA.PLANTED', 'AREA.HARVESTED', 'PRODUCTION', 'Longitude', 'Latitude' ]  # Add other yearly columns as needed

# =============================================================================
# def build_sequences(df, weekly_columns, yearly_columns, max_sequence_length=38):
#     sequences = []
#     targets = []
#     sequence_years = []  # List to track the year of each sequence
#     
#     for _, group in df.groupby(['county_name', 'year']):
#         sequence = group[weekly_columns + yearly_columns].values
#         if len(sequence) < max_sequence_length:
#             padding = np.zeros((max_sequence_length - len(sequence), len(weekly_columns) + len(yearly_columns)))
#             sequence = np.vstack((sequence, padding))
#         sequence = sequence[:max_sequence_length]  # Ensure all sequences are the same length
#         sequences.append(sequence)
#         targets.append(group['YIELD'].values[0])
#         sequence_years.append(group['year'].iloc[0])  # Assumes year is consistent within the group
#     
#     return np.array(sequences), np.array(targets), np.array(sequence_years)

# X, y, years = build_sequences(df, weekly_columns, yearly_columns)
# =============================================================================


def build_sequences(df, weekly_columns, yearly_columns, max_sequence_length=38):
    sequences = []
    targets = []
    sequence_years = []  # List to track the year of each sequence
    sequence_counties = []  # List to track the county of each sequence
    
    for _, group in df.groupby(['county_name', 'year']):
        sequence = group[weekly_columns + yearly_columns].values
        if len(sequence) < max_sequence_length:
            padding = np.zeros((max_sequence_length - len(sequence), len(weekly_columns) + len(yearly_columns)))
            sequence = np.vstack((sequence, padding))
        sequence = sequence[:max_sequence_length]  # Ensure all sequences are the same length
        sequences.append(sequence)
        targets.append(group['YIELD'].values[0])
        sequence_years.append(group['year'].iloc[0])
        sequence_counties.append(group['county_name'].iloc[0])
    
    return np.array(sequences), np.array(targets), np.array(sequence_years), np.array(sequence_counties)

X, y, years, counties = build_sequences(df, weekly_columns, yearly_columns)



# Determine the cutoff year for the split (e.g., 80% of the data for training)
cutoff_year = np.quantile(years, 0.8)

train_indices = years < cutoff_year
test_indices = years >= cutoff_year

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]


X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

print("NaNs in X_train:", np.isnan(X_train).any())
print("Infs in X_train:", np.isinf(X_train).any())
print("NaNs in y_train:", np.isnan(y_train).any())
print("Infs in y_train:", np.isinf(y_train).any())


from sklearn.preprocessing import StandardScaler, MinMaxScaler


scaler = StandardScaler()

nsamples, nx, ny = X_train.shape
nsamplest ,nxt, nyt = X_test.shape
X_train_reshaped = X_train.reshape((nsamples, nx*ny))
X_test_reshaped = X_test.reshape((nsamplest, nxt*nyt))

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train_reshaped)
# Use the same scaler to transform the test data
# Avoiding data leakage
X_test_scaled = scaler.transform(X_test_reshaped)

# Reshape back to original shape
X_train = X_train_scaled.reshape((nsamples, nx, ny))
X_test = X_test_scaled.reshape((nsamplest, nxt, nyt))

scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))

# Define the RNN model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=64)


# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE: {rmse}')

from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)

rmse_normalized = rmse / (y_test.max() - y_test.min())  # Normalize by range
print("Normalized RMSE:", rmse_normalized)


# Binning the data
bins = np.linspace(min(y_test), max(y_test), 4)  # Adjust the number of bins as necessary
y_test_binned = np.digitize(y_test, bins)
y_pred_binned = np.digitize(y_pred, bins)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_binned, y_pred_binned)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



######################################################################

# Example: Predict for the latest year and a specific county, e.g., 'Orange County'
latest_year = max(years[test_indices])  # Assuming test_indices are already defined
chosen_county = 'adair_iowa'

# Boolean indices for the year and county
latest_year_indices = (years == latest_year) & (counties == chosen_county) & test_indices

# Predict using the model for this specific county and year
X_latest_year = X[latest_year_indices]
y_latest_year = y[latest_year_indices]

# Ensure there is data for this county and year
if X_latest_year.size > 0:
    y_pred_latest_year = model.predict(X_latest_year)
    # Calculate RMSE for this specific county and year
    rmse_latest_year = np.sqrt(mean_squared_error(y_latest_year, y_pred_latest_year))
    print(f"RMSE for {chosen_county} in the year {latest_year}: {rmse_latest_year}")
else:
    print(f"No data available for {chosen_county} in the year {latest_year}.")
y_pred_rescaled = scaler.inverse_transform(y_pred_latest_year)
