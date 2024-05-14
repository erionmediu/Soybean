# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:28:23 2024

@author: user
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
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
df = pd.read_csv(join('writen data and checkpoints','df_finalv3_daily.csv'))
df = df[df["county_name"] != "massac_illinois"]  
df = df.drop('state_name', axis = 1)

df.info()
df.describe()
df.columns


df = df.sort_values(by=[ 'county_name', 'year', 'week', 'day'])
# Example of feature columns
weekly_columns = ['weekly_dayl', 'weekly_prcp', 'weekly_srad', 'weekly_swe', 'weekly_tmax', 'weekly_tmin', 'weekly_vp']
yearly_columns = ['AREA.PLANTED', 'AREA.HARVESTED', 'PRODUCTION', 'Longitude', 'Latitude']

# Define function to build sequences
def build_sequences(df, weekly_columns, yearly_columns, max_sequence_length=34):
    sequences = []
    targets = []
    sequence_years = []
    sequence_counties = []
    yearly_data = []

    for _, group in df.groupby(['county_name', 'year']):
        sequence = group[weekly_columns].values
        if len(sequence) < max_sequence_length:
            padding = np.zeros((max_sequence_length - len(sequence), len(weekly_columns)))
            sequence = np.vstack((sequence, padding))
        elif len(sequence) > max_sequence_length:
            sequence = sequence[:max_sequence_length]
            
        sequences.append(sequence)
        targets.append(group['YIELD'].iloc[0])
        sequence_years.append(group['year'].iloc[0])
        sequence_counties.append(group['county_name'].iloc[0])
        yearly_data.append(group[yearly_columns].iloc[0].values)
    
    return (np.array(sequences), 
            np.array(yearly_data), 
            np.array(targets), 
            np.array(sequence_years), 
            np.array(sequence_counties))

# Split the data into training, validation, and testing sets
train_df = df[df['year'] <= 2015]
validate_df = df[(df['year'] > 2015) & (df['year'] <= 2020)]
test_df = df[df['year'] > 2020]

# Apply scaling
scaler = StandardScaler()
scaler.fit(train_df[weekly_columns + yearly_columns])

train_df[weekly_columns + yearly_columns] = scaler.transform(train_df[weekly_columns + yearly_columns])
validate_df[weekly_columns + yearly_columns] = scaler.transform(validate_df[weekly_columns + yearly_columns])
test_df[weekly_columns + yearly_columns] = scaler.transform(test_df[weekly_columns + yearly_columns])

# Build sequences for each set
X_train, yearly_train, y_train, train_years, train_counties = build_sequences(train_df, weekly_columns, yearly_columns)
X_validate, yearly_validate, y_validate, validate_years, validate_counties = build_sequences(validate_df, weekly_columns, yearly_columns)
X_test, yearly_test, y_test, test_years, test_counties = build_sequences(test_df, weekly_columns, yearly_columns)





from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

# Define the LSTM input layer
weekly_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name='Weekly_Input')
x = LSTM(50, activation='relu', return_sequences=True)(weekly_input)
x = Dropout(0.2)(x)
x = LSTM(40, activation='relu', return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(30, activation='relu', return_sequences=False)(x)
x = Dropout(0.3)(x)

# Define the yearly input layer
yearly_input = Input(shape=(yearly_train.shape[1],), name='Yearly_Input')
y = Dense(10, activation='relu')(yearly_input)
y = Dropout(0.2)(y)

# Concatenate LSTM output with yearly features
combined = Concatenate()([x, y])

# Final dense layers
z = Dense(20, activation='relu')(combined)
z = Dropout(0.2)(z)
z = Dense(1)(z)

# Define the model
model = Model(inputs=[weekly_input, yearly_input], outputs=z)
model.compile(optimizer='adam', loss='mse')

# Summary of the model
model.summary()

# Train the model
history = model.fit(
    [X_train, yearly_train], y_train, 
    epochs=10, validation_data=([X_validate, yearly_validate], y_validate), 
    batch_size=64
)

# Evaluate the model
y_pred = model.predict([X_test, yearly_test])
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")
