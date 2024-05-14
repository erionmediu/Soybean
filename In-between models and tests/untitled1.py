import os
import pandas as pd
from os.path import join
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Change the working directory
os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")

# Load the data
df = pd.read_csv(join('writen data and checkpoints','df_finalv1.csv'))
df = df[df["county_name"] != "massac_illinois"]  # Filter out specific county data
df.info()
df.columns
# Filter data by weeks
df_filtered = df[(df['week'] >= 1) & (df['week'] <= 52)]
df_sorted = df_filtered.sort_values(by=['state_name', 'county_name', 'year', 'week'])

# Define columns
weekly_columns = ['weekly_prcp', 'weekly_srad', 'weekly_tmax', 'weekly_tmin', 'weekly_vp', 'percentage', 'percentage_overlap']
yearly_columns = ['AREA.PLANTED', 'AREA.HARVESTED', 'PRODUCTION', 'condition_score' ,'Longitude', 'Latitude']

# Function to build sequences with padding
def build_sequences(df, weekly_columns, yearly_columns, max_sequence_length=38):
    sequences = []
    targets = []
    sequence_years = []
    sequence_counties = []
    
    for _, group in df.groupby(['county_name', 'year']):
        sequence = group[weekly_columns + yearly_columns].values
        if len(sequence) < max_sequence_length:
            padding = np.zeros((max_sequence_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack((sequence, padding))
        sequences.append(sequence[:max_sequence_length])
        targets.append(group['YIELD'].values[0])
        sequence_years.append(group['year'].iloc[0])
        sequence_counties.append(group['county_name'].iloc[0])
    
    return np.array(sequences), np.array(targets), np.array(sequence_years), np.array(sequence_counties)

X, y, years, counties = build_sequences(df_sorted, weekly_columns, yearly_columns)

# Split data by year
cutoff_year = np.quantile(years, 0.8)
train_indices = years < cutoff_year
test_indices = years >= cutoff_year

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Convert to appropriate data type
X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

print(np.mean(y_train))
print(np.mean(y_test))
# Scaling the features
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
print(np.mean(y_train))
print(np.mean(y_test))

# Define and compile the LSTM model
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(nx, ny)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=64)

# Predict and inverse transform predictions
y_pred = model.predict(X_test)


# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE: {rmse}')

r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)

rmse_normalized = rmse / (y_test.max() - y_test.min())
print("Normalized RMSE:", rmse_normalized)

# Example prediction for a specific county and year
latest_year = max(years[test_indices])
chosen_county = 'adair_iowa'

latest_year_indices = (years == latest_year) & (counties == chosen_county) & test_indices
X_latest_year = X[latest_year_indices]
y_latest_year = y[latest_year_indices]

if X_latest_year.size > 0:
    y_pred = model.predict(X_latest_year)
    rmse_latest_year = np.sqrt(mean_squared_error(y_latest_year, y_pred))
    print(f"RMSE for {chosen_county} in the year {latest_year}: {rmse_latest_year}")
else:
    print(f"No data available for {chosen_county} in the year {latest_year}.")



