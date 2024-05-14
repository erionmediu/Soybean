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
import warnings

warnings.filterwarnings('ignore')

# Change the working directory
os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")

# Load the data
df = pd.read_csv(join('writen data and checkpoints','df_finalv3_daily.csv'))
df = df[df["county_name"] != "massac_illinois"]  
df = df.drop('state_name', axis = 1)


# Example of feature columns
daily_columns = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp' ]
weekly_columns = ['percentage', 'percentage_overlap', 'condition_score',] # removed for now 'stage' 'overlapping' 'condition_class'
yearly_columns = ['AREA.PLANTED', 'AREA.HARVESTED', 'PRODUCTION', 'Longitude', 'Latitude']


# Prepare data by extending weekly and yearly data
df['week_start'] = df['day'] - df['day'].mod(7)
weekly_data = df.groupby(['county_name', 'year', 'week_start'])[weekly_columns].transform('first')
df.update(weekly_data)

yearly_data = df.groupby(['county_name', 'year'])[yearly_columns].transform('first')
df.update(yearly_data)

# Sort data
df.sort_values(by=['county_name', 'year', 'day'], inplace=True)

# Split data into training, validation, and test sets
train_df = df[df['year'] <= 2015]
validate_df = df[(df['year'] > 2015) & (df['year'] <= 2020)]
test_df = df[df['year'] > 2020]

# Scaling
scaler = StandardScaler()
train_features = train_df[daily_columns + weekly_columns + yearly_columns]
scaler.fit(train_features)  # Fit only on training data

# Apply scaling
train_df[daily_columns + weekly_columns + yearly_columns] = scaler.transform(train_features)
validate_df[daily_columns + weekly_columns + yearly_columns] = scaler.transform(validate_df[daily_columns + weekly_columns + yearly_columns])
test_df[daily_columns + weekly_columns + yearly_columns] = scaler.transform(test_df[daily_columns + weekly_columns + yearly_columns])

# Build sequences (example function, adjust accordingly)
def build_sequences(df, max_days=365):
    sequences = []
    targets = []
    for _, group in df.groupby(['county_name', 'year']):
        tensor = group[daily_columns + weekly_columns + yearly_columns].values
        if tensor.shape[0] < max_days:
            padding = np.zeros((max_days - tensor.shape[0], tensor.shape[1]))
            tensor = np.vstack([tensor, padding])
        sequences.append(tensor[:max_days])
        targets.append(group['YIELD'].iloc[0])  # Assuming YIELD is the target

    return np.array(sequences), np.array(targets)

# Build tensors for each set
X_train, y_train = build_sequences(train_df)
X_validate, y_validate = build_sequences(validate_df)
X_test, y_test = build_sequences(test_df)

# Checking shapes
print(X_train.shape)


