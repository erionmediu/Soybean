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
# Step 1: Create a unique day index from the 'year' and 'week_start' columns
# First, ensure week_start is converted properly if it represents the day of the year
df['date'] = pd.to_datetime(df['year'].astype(str), format='%Y') + pd.to_timedelta(df['week_start'] - 1, unit='d')

# Sort and reset index to ensure continuity
df.sort_values(by='date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Create a Day_index column
start_date = df['date'].min()
df['Day_index'] = (df['date'] - start_date).dt.days

# Step 2: Drop unnecessary columns
df.drop(['year', 'week_start'], axis=1, inplace=True)

# Step 3: Optionally, you can drop other columns if they are not required
df.drop(['county_name', 'stage', 'overlapping' ,'condition_class'], axis=1, inplace=True)

# Now your DataFrame will have a continuous day index and maintain latitude and longitude for location,
# with yield and other variables as features for each day.

from sklearn.model_selection import train_test_split

# Assuming 'df' is your final DataFrame and 'YIELD' is the target variable
X = df.drop('YIELD', axis=1)  # Features
X = df.drop(['YIELD', 'date'], axis=1)  # Assuming 'date' is still in your dataframe

y = df['YIELD']  # Target variable

# Split the data into training and test sets (e.g., 80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# First, sort the actual values for a more interpretable plot
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



































import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set the working directory
os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")

# Load the data
df = pd.read_csv(join('writen data and checkpoints', 'df_finalv3_daily.csv'))
df = df[df["county_name"] != "massac_illinois"]  
df = df.drop('state_name', axis=1)

# Convert 'week_start' to a proper date and sort the DataFrame
df['date'] = pd.to_datetime(df['year'].astype(str), format='%Y') + pd.to_timedelta(df['day'] - 1, unit='d')
df.sort_values(by='date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Create a Day_index column
start_date = df['date'].min()
df['Day_index'] = (df['date'] - start_date).dt.days

# Exclude the county name for modeling, keep it for plotting
county_names = df['county_name']
df.drop(['county_name', 'year', 'day', 'date'], axis=1, inplace=True)
# Step 3: Optionally, you can drop other columns if they are not required
df.drop(['county_name', 'stage', 'overlapping' ,'condition_class'], axis=1, inplace=True)

# Define features and target variable
X = df.drop('YIELD', axis=1)
y = df['YIELD']

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
counties_test = county_names[X_test.index]  # Keep county names for test set

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Optional: Plot results for a specific county
specific_county = 'specific_county_name'
county_filter = counties_test == specific_county
y_test_county = y_test[county_filter]
y_pred_county = y_pred[county_filter]

plt.figure(figsize=(10, 6))
plt.scatter(y_test_county, y_pred_county, alpha=0.5)
plt.title(f'True vs. Predicted Yields for {specific_county}')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.plot([y_test_county.min(), y_test_county.max()], [y_test_county.min(), y_test_county.max()], 'k--', lw=4)
plt.show()

























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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# Set the working directory
os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")

# Load the data
df = pd.read_csv(join('writen data and checkpoints', 'df_finalv3_daily.csv'))
df = df[df["county_name"] != "massac_illinois"]
df = df[df['county_name']=="adams_illinois"]

df = df.drop('state_name', axis=1)

# Convert 'week_start' to a proper date and sort the DataFrame
df['date'] = pd.to_datetime(df['year'].astype(str), format='%Y') + pd.to_timedelta(df['day'] - 2, unit='d')
df.sort_values(by='date', inplace=True)
df.reset_index(drop=True, inplace=True)


# to change the split and check index at each desired time point :
# =============================================================================
# first_date_after_2018 = df[df['date'] >= '2018-01-01']['date'].min()
# print("First available date after 2018-01-01:", first_date_after_2018)
# =============================================================================

# Create a Day_index column
start_date = df['date'].min()
df['Day_index'] = (df['date'] - start_date).dt.days
mapping_df = df[['county_name', 'Longitude', 'Latitude', 'Day_index', 'date', 'year']]

split_start_2018 = df[df['date'] == '2018-04-21']['Day_index'].values[0]
split_end_2020 = df[df['date'] == '2018-11-30']['Day_index'].values[0]

# Split the data into training, validation, and test sets
train_df = df[df['Day_index'] < split_start_2018]
validation_df = df[(df['Day_index'] >= split_start_2018) & (df['Day_index'] <= split_end_2020)]
test_df = df[df['Day_index'] > split_end_2020]

# Prepare feature and target matrices for each set
droplist = ['county_name', 'stage', 'overlapping', 'condition_class', 'date', 'Day_index', 'YIELD' , 'year', 'day']
# think later of you can use this for example devide days into categories or week to make stages and more
X_train = train_df.drop(droplist, axis=1)
y_train = train_df['YIELD']
X_validation = validation_df.drop(droplist, axis=1)
y_validation = validation_df['YIELD']
X_test = test_df.drop(droplist, axis=1)
y_test = test_df['YIELD']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Scale features
scaler = StandardScaler()

columns_to_scale = [col for col in X_train.columns if col not in ['Longitude', 'Latitude']]

# Set up the scaler using ColumnTransformer to scale only specified columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columns_to_scale)
    ],
    remainder='passthrough'  # This leaves 'Longitude' and 'Latitude' untouched
)

# Apply the transformations
X_train_scaled = preprocessor.fit_transform(X_train)
X_validation_scaled = preprocessor.transform(X_validation)
X_test_scaled = preprocessor.transform(X_test)

###############################################################################
# =============================================================================
# # Train a linear regression model
# #model = LinearRegression()
# #model.fit(X_train_scaled, y_train)
# 
# 
# 
# from sklearn.model_selection import GridSearchCV, LeaveOneOut
# from xgboost import XGBRegressor
# 
# # Define the model
# model = XGBRegressor(objective='reg:squarederror')
# 
# # Parameters grid
# param_grid = {
#     'colsample_bytree': [0.3],
#     'learning_rate': [0.01],
#     'max_depth': [5],
#     'alpha': [5],
#     'n_estimators': [50]
# }
# 
# # Setup the LOOCV
# loo = LeaveOneOut()
# 
# # Setup the grid search
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=loo, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# 
# # Fit grid search
# grid_search.fit(X_train_scaled, y_train)
# 
# # Best model results
# print("Best parameters:", grid_search.best_params_)
# print("Best score (MSE):", -grid_search.best_score_)
# 
# 
# 
# 
# 
# =============================================================================



############## ABOVE NEEDS TO BE FIXED FOR CROSS VALIDATION
############# MAYBE AGREGATION CAN BE TESTED OUT ALL IN ONE NUMBER

model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.01,
                max_depth = 10, alpha = 10, n_estimators = 150)

model.fit(X_train_scaled, y_train)





##################################################################################

# Make predictions
y_pred = model.predict(X_test_scaled)


# Extract the mapping information for the test dataset
test_mapping_df = mapping_df[mapping_df['Day_index'].isin(test_df['Day_index'])]


# Create a DataFrame of predictions
predictions_df = pd.DataFrame(y_pred, columns=['Predicted_YIELD'])

# Reset index in test_mapping_df to concatenate properly
test_mapping_df.reset_index(drop=True, inplace=True)

# Concatenate predictions with the mapping information
final_predictions_df = pd.concat([test_mapping_df, predictions_df, y_test.reset_index(drop=True)], axis=1)
final_predictions_df.rename(columns={'YIELD': 'Actual_YIELD'}, inplace=True)
# Display the final DataFrame
print(final_predictions_df)


# Sort the DataFrame by county and year to ensure proper grouping
prediction_sorted = final_predictions_df.sort_values(by=['county_name', 'year'])

# Group by county and year, then aggregate the predicted yield using mean
annual_predictions = prediction_sorted.groupby(['county_name', 'year'])['Predicted_YIELD'].mean().reset_index()

# Get the first actual yield for each county-year combination
# Since all actual yields for a given year-county are the same, taking the first one is sufficient
annual_predictions['Actual_YIELD'] = prediction_sorted.groupby(['county_name', 'year'])['Actual_YIELD'].first().values

# Calculate and print the Mean Squared Error
mse = mean_squared_error(annual_predictions['Actual_YIELD'], annual_predictions['Predicted_YIELD'])
print(f'Mean Squared Error: {mse}')


y_test_county = annual_predictions['Actual_YIELD']
y_pred_county = annual_predictions['Predicted_YIELD']

plt.figure(figsize=(10, 6))
plt.scatter(y_test_county, y_pred_county, alpha=0.5)
plt.title('True vs. Predicted Yields for ')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.plot([y_test_county.min(), y_test_county.max()], [y_test_county.min(), y_test_county.max()], 'k--', lw=4)
plt.show()














































###########################################################

import hashlib

def hash_sequence(values):
    """Create a hash for a sequence of values."""
    # Convert values to a single string and hash it
    hash_object = hashlib.sha256(''.join(map(str, values)).encode())
    return hash_object.hexdigest()

# Group by county and year, apply hashing to each group's daily data
df = df[df['county_name']=="adams_illinois"]
OBJ = ['county_name','YIELD', 'year', 'tmax']
df = df[OBJ]
df['hash'] = df.groupby(['county_name', 'year'])['tmax'].transform(hash_sequence)
OBJ = ['county_name','YIELD', 'year', 'hash']
df = df[OBJ]
print(len(df['hash'].unique()))

df = df.drop_duplicates()

# THE HASHING TRICK ? 
# =============================================================================
# Now this must be implemented for every year in every variable that has such sequence than we end up in 
# yearly , only question remains how now do you use hash codes in a model to predict ?
# =============================================================================

#######################################################################################################
# Auto encoder approach 
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.models import Model




# Set the working directory
os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")

# Load the data
df = pd.read_csv(join('writen data and checkpoints', 'df_finalv3_daily.csv'))
df = df[df["county_name"] == "adams_illinois"]
OBJ = ['county_name','YIELD', 'year', 'tmax' , 'day']
df = df[OBJ]
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['day'].astype(str), format='%Y-%j')
df.sort_values(by='date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Check for NaNs and Infs
assert df['tmax'].isnull().sum() == 0, "There are NaN values in 'tmax'"
assert np.isfinite(df['tmax']).all(), "There are infinite values in 'tmax'"

# Normalize the 'tmax' values
scaler = StandardScaler()
df['tmax_normalized'] = scaler.fit_transform(df[['tmax']])

# Group data by year and create sequences
df['year'] = df['date'].dt.year
grouped = df.groupby('year')
input_seqs = []
for _, group in grouped:
    daily_temps = group['tmax_normalized'].values
    input_seqs.append(daily_temps)

# Pad sequences to have the same length
input_seqs = pad_sequences(input_seqs, padding='post', dtype='float32')

# Reshape for LSTM input
input_seqs = np.array(input_seqs)
input_seqs = np.expand_dims(input_seqs, axis=2)  # Assuming temperature is the only feature

# Split data
X_train, X_test = train_test_split(input_seqs, test_size=0.2, random_state=42)

# Define LSTM Autoencoder
input_dim = 1  # number of features per timestep
timesteps = X_train.shape[1]  # number of timesteps

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(100, activation='tanh')(inputs)
encoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(100, activation='tanh', return_sequences=True)(encoded)
decoded = TimeDistributed(Dense(input_dim))(decoded)

autoencoder = Model(inputs, decoded)
optimizer = Adam(learning_rate=0.001, clipnorm=1.0) 
autoencoder.compile(optimizer=optimizer, loss='mse')

# Train the autoencoder
history = autoencoder.fit(X_train, X_train, epochs=200, batch_size=128, validation_split=0.2)

# Encoding part
encoder = Model(inputs, encoded)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Ensure the embeddings are correctly shaped
print("X_train_encoded shape:", X_train_encoded.shape)
print("X_test_encoded shape:", X_test_encoded.shape)



# Prepare year and county mapping from original DataFrame
year_county = df[['year', 'county_name', 'YIELD']].drop_duplicates().reset_index(drop=True)

# Split the year_county similarly as you split your input sequences (ensure that they match the train/test split)
# This example assumes the split is directly correlated with how `X_train` and `X_test` were formed.
year_county_train, year_county_test = train_test_split(year_county, test_size=0.2, random_state=42)

# Combine the encoded features with the corresponding year, county, latitude, longitude, and yield
train_encoded_df = pd.DataFrame(X_train_encoded.reshape(X_train_encoded.shape[0], -1))  # Flatten encoded features
train_encoded_df = pd.concat([year_county_train.reset_index(drop=True), train_encoded_df], axis=1)

test_encoded_df = pd.DataFrame(X_test_encoded.reshape(X_test_encoded.shape[0], -1))  # Flatten encoded features
test_encoded_df = pd.concat([year_county_test.reset_index(drop=True), test_encoded_df], axis=1)

# Combine train and test back to a full dataset if needed
full_encoded_df = pd.concat([train_encoded_df, test_encoded_df], axis=0)

# Optionally, save this dataframe to a CSV for further use
full_encoded_df.to_csv('encoded_yearly_data.csv', index=False)

# Now full_encoded_df contains the encoded representation along with year, county, latitude, longitude, and yield
print(full_encoded_df.head())
print(full_encoded_df.shape)
