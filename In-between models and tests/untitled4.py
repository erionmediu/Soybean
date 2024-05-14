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


# Example of feature columns
daily_columns = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp' ]
weekly_columns = ['percentage', 'percentage_overlap', 'condition_score',] # removed for now 'stage' 'overlapping' 'condition_class'
yearly_columns = ['AREA.PLANTED', 'AREA.HARVESTED', 'PRODUCTION', 'Longitude', 'Latitude']



df['week_start'] = df['day'] - df['day'].mod(7)
weekly_data = df.groupby(['county_name', 'year', 'week_start'])[weekly_columns].transform('first')
for col in weekly_columns:
    df[col] = weekly_data[col]

# Extend yearly data to daily
yearly_data = df.groupby(['county_name', 'year'])[yearly_columns].transform('first')
df.update(yearly_data)

# Now all data in `df` is extended properly and can be reshaped into a 3D tensor
# Building the 3D tensor
# We will have one tensor per sample (i.e., per county per year)
unique_years_counties = df[['county_name', 'year']].drop_duplicates()

tensors = []
for _, row in unique_years_counties.iterrows():
    sample_df = df[(df['county_name'] == row['county_name']) & (df['year'] == row['year'])]
    tensor = sample_df.sort_values(by='day')[daily_columns + weekly_columns + yearly_columns].to_numpy()
    # Ensure the tensor has the same length for all samples, padding if necessary
    max_days = 365  # or 366 for leap years if applicable
    if tensor.shape[0] < max_days:
        padding = np.zeros((max_days - tensor.shape[0], tensor.shape[1]))
        tensor = np.vstack([tensor, padding])
    tensors.append(tensor[:max_days])

# `tensors` is now a list of 3D arrays where each array is (time_steps, features)
# Convert list to a numpy array for processing with machine learning models
tensors = np.array(tensors)

print(tensors.shape)
