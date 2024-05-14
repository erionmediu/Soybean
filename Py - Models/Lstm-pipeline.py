from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from os.path import join
import pandas as pd 
import numpy as np
# =============================================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
# =============================================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
# =============================================================================
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os.path 

'''

Implementation works on one county now it must be tested in multiple counties
+ data needs cleaning at a weekly level since after weeks go to 100 percent
they move on this my be R problem
+ weeks are important for the padding part
+ there are several variables that are left out this variables must be re -
considered.
+ Than this pipeline is complete
+ Next remains the other 2 piplines or approaches to model that data.
'''

 

# =============================================================================

os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")
df = pd.read_csv(join('writen data and checkpoints','df_finalv3_daily.csv'))

# =============================================================================


# df = pd.read_csv('C:/Users/Mediue/OneDrive - Allianz/Desktop/sample_data_test.csv')
df = df[df["county_name"]!= "massac_illinois"]

# MASACC ILLINOIS TMAX DATA IN 1980
# MUST BE RE DOWNLOADED THIS MEANS --> ORCHESTRATE DOWNLOAD AND PROCCESS NC FILES
# test = df[df['weekly_tmax'].isna()== True]

daily_columns = ['prcp', 'srad', 'tmax', 'tmin', 'vp' , 'dayl'] # swe out for now
weekly_columns = ['percentage' ,'percentage_overlap', 'condition_score'] # 'condition_class', 'overlapping']
yearly_columns = ['YIELD','AREA.PLANTED', 'AREA.HARVESTED', 'PRODUCTION', 'Longitude', 'Latitude']

# YIELD IS THE TARGET <-------------------------------------------------------------------------

'''

Explore the distribution of the weeks :
    What is the min / max week.
    What is the distribution/number above min and max.
    REMAINS MORE DIFFICULT THAT INITIALLY THOUGHT IT MAY NEED CAREFUL CONSIDERATION
    IN RELATION TO MAKE THIS WEEK INDEXES MATCH FOR ALL THE PROBLEM IS THAT PADDING
    IS EXTENSIVE BUT IF THERE IS A LAYER WHICH CAN IGNORE PADING MAYBE THIS CAN BE IGNORED TO
'''


conclusion_explored_weekA = 14 # To be identified
conclusion_explored_weekB = 52 # To be identified

df_filtered = df[(df['week'] >= conclusion_explored_weekA) & (df['week'] <= conclusion_explored_weekB)]
df_sorted = df_filtered.sort_values(by=[ 'county_name', 'year', 'week', 'day'])

'''We are using weekly to create the sequence since in this lSTM integration we are
   using weekly data , if this is chosen path this should be done daily to gain more data'''

def build_daily_sequences(df, daily_columns, weekly_columns, yearly_columns, sequence_length=38):
    # Lists to store the sequences, targets, sequence metadata (year and county)
    sequences = []
    targets = []
    sequence_years = []
    sequence_counties = []
    i = 0

    # Group data by county and year to create sequences for each unique pair
    for (_, group) in df.groupby(['county_name', 'year']):
        # Ensure group is sorted by day to maintain chronological order
        group = group.sort_values(by='day')
        # Extract the yearly data which remains constant for the group
        yearly_data = group[yearly_columns].iloc[0].tolist()  # Only need to grab the first since it's constant
        # Initialize empty data structure for sequence
        sequence = []
        print(i)
        i+=1


        # Process each day's data
        for _, row in group.iterrows():
            # Combine daily data with the constant yearly data for that year and county
            day_data = row[daily_columns].tolist() + yearly_data
            sequence.append(day_data)

        # If the actual sequence is shorter than desired, pad it with zeros
        if len(sequence) < sequence_length:
            padding_size = sequence_length - len(sequence)
            total_features = len(daily_columns) + len(yearly_columns)
            padding = [np.zeros(total_features) for _ in range(padding_size)]
            sequence.extend(padding)

    
        # Truncate to the desired sequence length if necessary
        sequence = sequence[:sequence_length]

        # Append to the lists
        sequences.append(sequence)
        targets.append(group['YIELD'].iloc[-1])  # Target is the yield of the last entry
        sequence_years.append(group['year'].iloc[0])
        sequence_counties.append(group['county_name'].iloc[0])

    # Convert lists to numpy arrays for compatibility with modeling tools
    return np.array(sequences), np.array(targets), np.array(sequence_years), np.array(sequence_counties)

 
# Use the function
X, y, years, counties = build_daily_sequences(df, daily_columns, weekly_columns, yearly_columns)
print(len(df['year'].unique()))

# Determine the cutoff year <-------
cutoff_year = np.quantile(years, 0.8)
print(cutoff_year)

train_indices = years < cutoff_year
test_indices = years >= cutoff_year

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

 

# Final check fo NA

print("NaNs in X_train:", np.isnan(X_train).any())
print("Infs in X_train:", np.isinf(X_train).any())
print("NaNs in y_train:", np.isnan(y_train).any())
print("Infs in y_train:", np.isinf(y_train).any())

#

 

 

# Assume we already have X_train and X_test which include daily and yearly columns

daily_features_count = len(daily_columns)  # Number of daily columns
yearly_features_count = len(yearly_columns) - 2  # Excluding Longitude and Latitude

 

# Calculate the total number of features to scale
features_to_scale = daily_features_count + yearly_features_count

# Reshape data for scaling
nsamples, nx, ny = X_train.shape
nsamplest ,nxt, nyt = X_test.shape
X_train_reshaped = X_train.reshape((nsamples * nx, ny))
X_test_reshaped = X_test.reshape((nsamplest, nxt*nyt))

 

# Indices of Longitude and Latitude in your yearly_columns

longitude_idx = yearly_columns.index('Longitude') + daily_features_count
latitude_idx = yearly_columns.index('Latitude') + daily_features_count

 

# Select columns to scale (all but Longitude and Latitude)
columns_to_scale = list(range(daily_features_count)) + list(range(daily_features_count, ny))
columns_to_scale.remove(longitude_idx)
columns_to_scale.remove(latitude_idx)

 

# Initialize scaler
scaler = StandardScaler()

 

# Apply scaler to the appropriate columns

X_train_scaled = np.copy(X_train_reshaped)
X_test_scaled = np.copy(X_test_reshaped)
X_train_scaled[:, columns_to_scale] = scaler.fit_transform(X_train_reshaped[:, columns_to_scale])
X_test_scaled[:, columns_to_scale] = scaler.transform(X_test_reshaped[:, columns_to_scale])

 

# Reshape back to original shape

X_train_scaled = X_train_scaled.reshape((nsamples, nx, ny))
X_test_scaled = X_test_scaled.reshape((nsamplest, nxt, nyt))

 
from tensorflow.keras.layers import TimeDistributed
 
from tensorflow.keras.layers import TimeDistributed

model = Sequential([
    Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])),  # Masking layer to ignore padding
    LSTM(50, activation='tanh', return_sequences=True),  # First LSTM layer
    Dropout(0.2),  # Dropout for the first LSTM layer
    LSTM(50, activation='tanh', return_sequences=True),  # Second LSTM layer
    Dropout(0.2),  # Dropout for the second LSTM layer
    TimeDistributed(Dense(100, activation='tanh')),  # Dense layer applied to each LSTM output timestep
    Dropout(0.3),  # Dropout after Dense layer
    TimeDistributed(Dense(50, activation='tanh')),  # Another TimeDistributed Dense layer
    Dropout(0.3),  # Dropout after second Dense layer
    TimeDistributed(Dense(1))  # Output layer applied to each timestep
])

# Using Adam optimizer with gradient clipping
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train,epochs=25, validation_data=(X_test, y_test), batch_size=64)


# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE: {rmse}')

rmse_normalized = rmse / (y_test.max() - y_test.min())  # Normalize by range
print("Normalized RMSE:", rmse_normalized)



# Scatter plot of actual vs. predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted')
plt.xlabel('Actual Yields')
plt.ylabel('Predicted Yields')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)  # Plot a diagonal line
plt.show()