
"""
Created on Sun May 26 18:46:36 2024

@author: Erion Mediu 

This code represents the code used for the derivation of the results of the 
thesis presented to tilburg university. The code relies on prior download
proccedures that were conducteed in R. Thus if interested in the data downlad
and pre-proccess proccedure please refer to the R code within this repository
"""



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from os.path import join
import pandas as pd 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os.path 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb

# Set working directory : 
    
os.chdir("C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r")

df = pd.read_csv(join('writen data and checkpoints','df_finalv_noengcond_clas.csv'))

obj = ['PCT.EXCELLENT', 'PCT.FAIR', 'PCT.GOOD', 'PCT.POOR', 'PCT.VERY.POOR']
df[obj] = df[obj].fillna(0)

df['year'].unique()

df = df[df["county_name"]!= "massac_illinois"] 

illinois_df_weth= pd.read_csv(join('writen data and checkpoints','weekly_Weather_Data_ILLINOIS.csv'))
illinois_df_weth.sort_values(by = ['week','county_name'])
iowa_df_weth= pd.read_csv(join('writen data and checkpoints','weekly_Weather_Data_iowa.csv'))
iowa_df_weth.sort_values(by = ['week','county_name'])
illinois_df_weth['state_name'] = "ILLINOIS"
iowa_df_weth['state_name'] = "IOWA"
combined_data = pd.concat([illinois_df_weth, iowa_df_weth], ignore_index=True)
combined_data['county_name'] = combined_data['county_name'].str.lower()
combined_data['county_name'] = combined_data.apply(
    lambda row: f"{row['county_name']}_{row['state_name'].lower()}"
    if row['state_name'] in ["ILLINOIS", "IOWA"]
    else row['county_name'],
    axis=1
)
print(combined_data.head())
combined_data = combined_data[combined_data["week"]!= 13] 
combined_data = combined_data[combined_data["week"]!= 1] 
combined_data = combined_data[combined_data["week"]!= 53] 
weekly_wather_col = ['weekly_dayl', 'weekly_prcp', 'weekly_srad',
       'weekly_tmax', 'weekly_tmin', 'weekly_vp' ]
yearly_col = ['year', 'state_name','county_name']

def weather_sorter(df = df ,weekly_transform_col = weekly_wather_col ,OBJ = yearly_col):
    df= df.sort_values(by=[ 'county_name', 'year', 'week'])

    df_wide = df.pivot_table(
        index=['year', 'county_name'], 
        columns='week', 
        values=weekly_transform_col
        )

    df_wide.columns = [f'{col[0]}_week{col[1]}' for col in df_wide.columns]

    df_wide.reset_index(inplace=True)
    df_main = df[OBJ].drop_duplicates()
    df_final = pd.merge(df_main, df_wide, on=['year', 'county_name'], how='left')
    df_final.info()
    return(df_final)


weather_wide =  weather_sorter(df = combined_data ,weekly_transform_col = weekly_wather_col ,OBJ = yearly_col)
  
# =============================================================================
yearly_col = ['YIELD','year', 'state_name','county_name','AREA.PLANTED',
              'PRODUCTION', 'AREA.HARVESTED' ,'Longitude' ,'Latitude' , 'PRODUCTION' , 'AREA.PLANTED']
df1 = df[yearly_col]
# =============================================================================

merged_df = pd.merge(weather_wide, df1, on=['county_name', 'year'], how='left')
OBJ=['county_name', 'year']
merged_df = merged_df.drop_duplicates()
merged_df = merged_df[merged_df['YIELD']!=0]

df_final = merged_df


OBJ = ['state_name_y', 'state_name_x' ]
df_final.drop(columns=OBJ, inplace=True)
df_final = df_final.dropna()

test = df_final[df_final['YIELD']==0]
test.shape
test = df_final[df_final['YIELD'].isna()]
test.shape


df_final.describe()

df_final.fillna(0, inplace=True)


df_final['year'].unique()


# Define non-weather variables
weekly_non_w = ['PCT.SETTING.PODS', 'PCT.HARVESTED', 'PCT.EXCELLENT', 'PCT.PLANTED', 'PCT.EMERGED', 'PCT.BLOOMING', 'PCT.SETTING.PODS', 'PCT.COLORING', 'PCT.DROPPING.LEAVES', 'PCT.HARVESTED']

def weekly_nonw(df, merger, weekly_non_w):
    def minmaxfinder(a, week='week', df=None):
        obj = [a, week]
        test = df[obj]
        test = test.dropna(subset=[a])
        test = test[test[a] != 0]
        maxi = test[week].max()
        mini = test[week].min()
        return mini, maxi

    week_ranges = {}
    for i in weekly_non_w:
        min_max = minmaxfinder(i, df=df)
        week_ranges[i] = min_max

    result_df = pd.DataFrame()
    for variable, (min_week, max_week) in week_ranges.items():
        df_filtered = df[(df['week'] >= min_week) & (df['week'] <= max_week)]
        pivot_df = df_filtered.pivot_table(index=['year', 'county_name'], columns='week', values=variable, fill_value=0)
        pivot_df.columns = [f'{variable}_week_{week}' for week in pivot_df.columns]
        pivot_df.reset_index(inplace=True)

        if result_df.empty:
            result_df = pivot_df
        else:
            result_df = pd.merge(result_df, pivot_df, on=['year', 'county_name'], how='outer')

    final_df = pd.merge(merger, result_df, on=['year', 'county_name'], how='left')
    return final_df

# Apply the transformation to non-weather variables
df_final = weekly_nonw(df, df_final, weekly_non_w)

# Clean up the dataframe by removing columns with only NaN values
def filter_and_clean(df):
    cols_with_only_na = df.columns[df.isna().all()].tolist()
    df.drop(columns=cols_with_only_na, inplace=True)
    print(cols_with_only_na)

filter_and_clean(df_final)

df_final.describe()

# Ensure no missing values
df_final.fillna(0, inplace=True)

for i in df_final.columns:
    print(i)

df_final['year'].unique()


# Plot if missing any :
    
plt.figure(figsize=(15, 10))
ax = sns.heatmap(df_final.isna(), cbar=True, cmap='viridis')
cbar = ax.collections[0].colorbar
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['No NA', 'NA'])
plt.title('Map of NA Values in df_final')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()



''' CLUSTER LOCATION AND WEATHER : Clustered on average weather per week acrross the years for a county 
so all weeks averaged accross year per county <---------------------------------------------------------------------------- '''

weather_features = ['weekly_dayl_', 'weekly_prcp_', 'weekly_srad_', 'weekly_tmax_', 'weekly_tmin_', 'weekly_vp_']

# Function to generate the column names dynamically
def generate_columns(base, weeks):
    return [f"{base}week{week}" for week in weeks]

# Weeks in the dataset
weeks = list(range(14, 53))

# Create a dictionary for aggregation
agg_dict = {}
for feature in weather_features:
    for week in weeks:
        column_name = f"{feature}week{week}"
        agg_dict[column_name] = ['mean', 'std']

# Aggregate the weather data
aggregated_data = df_final.groupby(['county_name']).agg(agg_dict).reset_index()

# Flatten the multi-level columns
aggregated_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in aggregated_data.columns.values]

# Rename columns to more meaningful names
renamed_columns = {}
for feature in weather_features:
    for week in weeks:
        renamed_columns[f"{feature}week{week}_mean"] = f"mean_{feature}week{week}"
        renamed_columns[f"{feature}week{week}_std"] = f"std_{feature}week{week}"

aggregated_data.rename(columns=renamed_columns, inplace=True)

print(aggregated_data.head())

# Merge aggregated data with location data
df_final_location = df_final[['county_name', 'Longitude', 'Latitude']].drop_duplicates()
combined_data = pd.merge(aggregated_data, df_final_location, on='county_name')

print(combined_data.head())

# Select features for clustering
features = [col for col in combined_data.columns if 'mean_' in col or 'std_' in col] + ['Longitude', 'Latitude']
scaler = StandardScaler()
combined_data_scaled = scaler.fit_transform(combined_data[features])

print(combined_data_scaled[:5])

# Apply KMeans clustering
n_clusters = 20  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
combined_data['cluster'] = kmeans.fit_predict(combined_data_scaled)

print(combined_data.head())
manual_colors = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
]

# Create a dictionary to map cluster IDs to these colors
color_map = {i: manual_colors[i] for i in range(20)}

# Visualize the clusters
plt.figure(figsize=(14, 10))
# Use the same colors for the same cluster IDs
for cluster_id in range(n_clusters):
    cluster_data = combined_data[combined_data['cluster'] == cluster_id]
    plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'],  color=color_map[cluster_id], alpha=0.6, edgecolors='w', s=100)

# Plot the cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, -2], centers[:, -1], c='black', s=300, alpha=0.8, marker='x', label='Centers')
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.title('County Clusters Based on Weather Conditions', fontsize=16)
plt.grid(False)
plt.show()

# Left join to assign clusters back to the original DataFrame
df_final= pd.merge(df_final, combined_data[['county_name', 'cluster']], on='county_name', how='left')

''' Yield for one cluster<---------------------------------------------------------------------------- '''

selected_cluster = 1  # You can change this to any cluster number you want to analyze

# Filter data for the selected cluster
cluster_data = df_final[df_final['cluster'] == selected_cluster]

# Plot yield for the selected cluster
plt.figure(figsize=(12, 8))

for county in cluster_data['county_name'].unique():
    county_data = cluster_data[cluster_data['county_name'] == county]
    plt.plot(county_data['year'], county_data['YIELD'], label=county, marker='o', linestyle='-')

plt.xlabel('Year')
plt.ylabel('Yield')
plt.title(f'Yield Over Time for Cluster {selected_cluster}')
plt.legend(loc='best', bbox_to_anchor=(1, 1), ncol=1)
plt.grid(False)
plt.show()


'''t - median '''
weeks = range(19, 35)

for week in weeks:
    df_final[f't_avg_week{week}'] = (df_final[f'weekly_tmax_week{week}'] + df_final[f'weekly_tmin_week{week}']) / 2
    df_final[f't_median_week{week}'] = df_final[[f'weekly_tmax_week{week}', f'weekly_tmin_week{week}']].median(axis=1)

'''mediian plot'''
# Extract the t-median columns and the yield column
t_median_columns = [f't_median_week{week}' for week in weeks]
correlation_data = df_final[t_median_columns + ['YIELD']]

# Compute the correlation matrix
correlation_matrix = correlation_data.corr()

# Plot the heatmap for the median correlation matrix
plt.figure(figsize=(20, 14))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Between T-Median (Week 19-34) and Yield', fontsize=20)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30)
plt.show()

'''mean plot'''

# Extract the t-median columns and the yield column
t_mean_columns = [f't_avg_week{week}' for week in weeks]
correlation_data = df_final[t_mean_columns + ['YIELD']]

# Compute the correlation matrix
correlation_matrix = correlation_data.corr()

# Plot the heatmap for the median correlation matrix
plt.figure(figsize=(20, 14))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Between average temp (Week 19-34) and Yield', fontsize=20)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30)
plt.show()



'''precipitation '''

# Define weeks of interest
weeks = range(19, 35)

# Extract the weekly precipitation columns and the yield column
prcp_columns = [f'weekly_prcp_week{week}' for week in weeks]
correlation_data = df_final[prcp_columns + ['YIELD']]

# Compute the correlation matrix
correlation_matrix = correlation_data.corr()

# Plot the heatmap
plt.figure(figsize=(20, 14))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Between Precipitation (Week 19-35) and Yield', fontsize = 20)

# Rotate x-ticks
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30)

plt.show()


# Matrix for other var 
""" Duplicates """

def remove_duplicate_and_sparse_columns(df, threshold=0.7):
    # Identify and remove duplicate columns
    duplicated_cols = df.columns[df.columns.duplicated()].tolist()
    print(f"Duplicate columns: {duplicated_cols}")
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Identify and remove columns with only 0s
    cols_with_only_zeros = df.columns[(df == 0).all()].tolist()
    print(f"Columns with only 0s: {cols_with_only_zeros}")
    df.drop(columns=cols_with_only_zeros, inplace=True)
    
    # Identify and remove columns with more than threshold% 0s
    cols_with_many_zeros = df.columns[(df == 0).mean() > threshold].tolist()
    print(f"Columns with more than {threshold*100}% 0s: {cols_with_many_zeros}")
    df.drop(columns=cols_with_many_zeros, inplace=True)
    
    return df

# Apply the function to your dataframe
df_final = remove_duplicate_and_sparse_columns(df_final)



obj = ['YIELD','year', 'AREA.PLANTED','PRODUCTION','AREA.HARVESTED','Longitude','Latitude', 'cluster']
df_corrplot = df_final[obj]
df_corrdf_corrplot = df_corrplot.drop_duplicates()


# Compute the correlation matrix
correlation_matrix = df_corrplot.corr()

# Plot the heatmap
plt.figure(figsize=(20, 14))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Between Precipitation (Week 19-35) and Yield', fontsize = 20)

# Rotate x-ticks
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30)

plt.show()

OBJ = ['county_name' ]
df_final.drop(columns=OBJ, inplace=True)


'''Basline OLS . normal time split , With PCA and withot '''

train_df = df_final[df_final['year'] <= 2019]
test_df = df_final[df_final['year'] >= 2022]

# Separate features and target variable
X_train = train_df.drop(columns=['YIELD', 'cluster'])
y_train = train_df['YIELD']

X_test = test_df.drop(columns=['YIELD', 'cluster'])
y_test = test_df['YIELD']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data
X_test_scaled = scaler.transform(X_test)
holder_best = {}

for i in range(1, 100):
     pca = PCA(n_components=i)
     X_train_pca = pca.fit_transform(X_train_scaled)
     X_test_pca = pca.transform(X_test_scaled)
     
     encoder = OneHotEncoder(sparse=False, drop='first')
     clusters_train_encoded = encoder.fit_transform(train_df[['cluster']])
     clusters_test_encoded = encoder.transform(test_df[['cluster']])

     # Combine PCA components with cluster encodings
     X_train_pca = np.hstack((X_train_pca, clusters_train_encoded))
     X_test_pca = np.hstack((X_test_pca, clusters_test_encoded))


     # Fit the linear regression model using the PCA components
     model = LinearRegression()
     model.fit(X_train_pca, y_train)
 
    # Make predictions
     y_train_pred = model.predict(X_train_pca)
     y_test_pred = model.predict(X_test_pca)
 
     # Evaluate the model
     train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
     test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
 
     train_r2 = r2_score(y_train, y_train_pred)
     test_r2 = r2_score(y_test, y_test_pred)
 
     #print(f'Train RMSE pca ols {i}: {train_rmse}, Train R^2: {train_r2}')
     print(f'Test RMSE pca ols {i}: {test_rmse}, Test R^2: {test_r2}')
     holder_best[i] = test_r2
     
# =============================================================================

# print(holder_best)
max_key = max(holder_best, key=holder_best.get)
# Get the maximum value
max_value = holder_best[max_key]
print(f"Key: {max_key}, Value: {max_value}")


test_rms = []
train_rms = []
train_rsq = []
test_rsq = []


# =============================================================================
for i in range(1, 100):
     pca = PCA(n_components=i)
     X_train_pca = pca.fit_transform(X_train_scaled)
     X_test_pca = pca.transform(X_test_scaled)
     
     encoder = OneHotEncoder(sparse=False, drop='first')
     clusters_train_encoded = encoder.fit_transform(train_df[['cluster']])
     clusters_test_encoded = encoder.transform(test_df[['cluster']])

     # Combine PCA components with cluster encodings
     X_train_pca = np.hstack((X_train_pca, clusters_train_encoded))
     X_test_pca = np.hstack((X_test_pca, clusters_test_encoded))


     # Fit the linear regression model using the PCA components
     model = LinearRegression()
     model.fit(X_train_pca, y_train)
 
    # Make predictions
     y_train_pred = model.predict(X_train_pca)
     y_test_pred = model.predict(X_test_pca)
 
     # Evaluate the model
     train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
     test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
 
     train_r2 = r2_score(y_train, y_train_pred)
     test_r2 = r2_score(y_test, y_test_pred)
 
     #print(f'Train RMSE pca ols {i}: {train_rmse}, Train R^2: {train_r2}')
     print(f'Test RMSE pca ols {i}: {test_rmse}, Test R^2: {test_r2}')
     holder_best[i] = test_r2
# =============================================================================

# print(holder_best)
max_key = max(holder_best, key=holder_best.get)
# Get the maximum value
max_value = holder_best[max_key]
print(f"Key: {max_key}, Value: {max_value}")

# Apply PCA to the scaled data
pca = PCA(n_components=max_key )  # Adjust the number of components based on your needs
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Encode the cluster column
encoder = OneHotEncoder(sparse=False, drop='first')
clusters_train_encoded = encoder.fit_transform(train_df[['cluster']])
clusters_test_encoded = encoder.transform(test_df[['cluster']])

# Combine PCA components with cluster encodings
X_train_final = np.hstack((X_train_pca, clusters_train_encoded))
X_test_final = np.hstack((X_test_pca, clusters_test_encoded))


# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(17, 11))

# Plotting
sns.histplot(test_rms, kde=True, ax=axes[0, 0], color='blue')
axes[0, 0].set_title('Test RMSE Distribution' , fontsize=25)

sns.histplot(train_rms, kde=True, ax=axes[0, 1], color='green')
axes[0, 1].set_title('Train RMSE Distribution' , fontsize=25)

sns.histplot(train_rsq, kde=True, ax=axes[1, 0], color='red')
axes[1, 0].set_title('Train R^2 Distribution', fontsize=25)

sns.histplot(test_rsq, kde=True, ax=axes[1, 1], color='purple')
axes[1, 1].set_title('Test R^2 Distribution', fontsize=25)

# Fine-tune layout
plt.tight_layout()
plt.show()



'''Random forest with extending window cv , without pca'''

# Function to perform moving forward extending window training and validation with 3-year increments
def moving_forward_validation(df, start_year, end_year, model, param_grid=None):
    # Initialize lists to accumulate results
    validation_years = []
    actual_yields = []
    predicted_yields = []
    rmse_scores = []
    r2_scores = []
    
    scaler = StandardScaler()
    
    # Perform the validation with a 3-year increment
    for val_year in range(start_year + 3, end_year, 3):
        train_years = range(start_year, val_year)
        
        train_df = df[df['year'].isin(train_years)]
        validation_df = df[df['year'] == val_year]
        
        X_train = train_df.drop(columns=['YIELD'])
        y_train = train_df['YIELD']
        X_validation = validation_df.drop(columns=['YIELD'])
        y_validation = validation_df['YIELD']
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_validation_scaled = scaler.transform(X_validation)
        
        # Hyperparameter tuning using GridSearchCV
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_validation_pred = best_model.predict(X_validation_scaled)
        validation_rmse = mean_squared_error(y_validation, y_validation_pred, squared=False)
        validation_r2 = r2_score(y_validation, y_validation_pred)
        
        print(f"Training up to year {val_year - 1} and validating on year {val_year}")
        print(f"Validation RMSE: {validation_rmse}, Validation R^2: {validation_r2}")
        
        # Accumulate results
        validation_years.extend([val_year] * len(y_validation))
        actual_yields.extend(y_validation)
        predicted_yields.extend(y_validation_pred)
        rmse_scores.append(validation_rmse)
        r2_scores.append(validation_r2)
    
    # Plot the accumulated results from validation
    plt.figure(figsize=(14, 8))
    plt.plot(validation_years, actual_yields, 'bo-', label='Actual Yield')
    plt.plot(validation_years, predicted_yields, 'ro-', label='Predicted Yield')
    plt.xlabel('Year')
    plt.ylabel('Yield')
    plt.title('Actual vs Predicted Yield Over Validation Years')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    return best_model

df = df_final

# Define the Random Forest model
model = RandomForestRegressor()

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 30, 50],
    'min_samples_split': [50, 100, 200],
    'min_samples_leaf': [1, 2, 4]
}

# Perform moving forward validation from 1980 to 2018, training up to 2018
best_model = moving_forward_validation(df, start_year=1980, end_year=2018, model=model, param_grid=param_grid)

# Final model evaluation on the test set without PCA
train_df = df_final[df_final['year'] <= 2018]
test_df = df_final[df_final['year'] >= 2019]

X_train = train_df.drop(columns=['YIELD'])
y_train = train_df['YIELD']
X_test = test_df.drop(columns=['YIELD'])
y_test = test_df['YIELD']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_model.fit(X_train_scaled, y_train)

y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Test RMSE without PCA: {test_rmse}, Test R^2: {test_r2}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, label='Test Data', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield (Without PCA)')
plt.legend()
plt.show()

# Find the optimal number of PCA components
holder_best = {}

for i in range(1, 101):
    pca = PCA(n_components=i)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    best_model.fit(X_train_pca, y_train)
    
    y_train_pred = best_model.predict(X_train_pca)
    y_test_pred = best_model.predict(X_test_pca)
    
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f'Test RMSE with PCA components {i}: {test_rmse}, Test R^2: {test_r2}')
    holder_best[i] = test_r2

max_key = max(holder_best, key=holder_best.get)
max_value = holder_best[max_key]
print(f"Best number of PCA components: {max_key} with R^2: {max_value}")

# Apply PCA to the scaled data with the best number of components
pca = PCA(n_components=max_key)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

best_model.fit(X_train_pca, y_train)

y_train_pred = best_model.predict(X_train_pca)
y_test_pred = best_model.predict(X_test_pca)

test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Test RMSE with optimal PCA components: {test_rmse}, Test R^2: {test_r2}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, label='Test Data', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield (With PCA)')
plt.legend()
plt.show()







'''XGBOOST with extending window cv , with AND without  pca'''

# Function to perform moving forward extending window training and validation with 3-year increments
def moving_forward_validation(df, start_year, end_year, model, param_grid=None):
    validation_years = []
    actual_yields = []
    predicted_yields = []
    rmse_scores = []
    r2_scores = []
    
    scaler = StandardScaler()
    
    for val_year in range(start_year + 3, end_year, 3):
        train_years = range(start_year, val_year)
        
        train_df = df[df['year'].isin(train_years)]
        validation_df = df[df['year'] == val_year]
        
        X_train = train_df.drop(columns=['YIELD'])
        y_train = train_df['YIELD']
        X_validation = validation_df.drop(columns=['YIELD'])
        y_validation = validation_df['YIELD']
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_validation_scaled = scaler.transform(X_validation)
        
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train_scaled, y_train)
        
        y_validation_pred = best_model.predict(X_validation_scaled)
        validation_rmse = mean_squared_error(y_validation, y_validation_pred, squared=False)
        validation_r2 = r2_score(y_validation, y_validation_pred)
        
        print(f"Training up to year {val_year - 1} and validating on year {val_year}")
        print(f"Validation RMSE: {validation_rmse}, Validation R^2: {validation_r2}")
        
        validation_years.extend([val_year] * len(y_validation))
        actual_yields.extend(y_validation)
        predicted_yields.extend(y_validation_pred)
        rmse_scores.append(validation_rmse)
        r2_scores.append(validation_r2)
    
    plt.figure(figsize=(14, 8))
    plt.plot(validation_years, actual_yields, 'bo-', label='Actual Yield')
    plt.plot(validation_years, predicted_yields, 'ro-', label='Predicted Yield')
    plt.xlabel('Year')
    plt.ylabel('Yield')
    plt.title('Actual vs Predicted Yield Over Validation Years')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(14, 8))
    plt.plot(range(start_year + 3, end_year, 3), rmse_scores, 'go-', label='Validation RMSE')
    plt.xlabel('Year')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE Over Years')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_model

df = df_final

# Define the XGBoost model
model = xgb.XGBRegressor()

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [5, 50, 100],
    'max_depth': [10, 30, 50],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.5, 0.7, 1.0]
}

# Perform moving forward validation from 1980 to 2018, training up to 2018
best_model = moving_forward_validation(df, start_year=1980, end_year=2018, model=model, param_grid=param_grid)

# Final model evaluation on the test set without PCA
train_df = df_final[df_final['year'] <= 2018]
test_df = df_final[df_final['year'] >= 2019]

X_train = train_df.drop(columns=['YIELD'])
y_train = train_df['YIELD']
X_test = test_df.drop(columns=['YIELD'])
y_test = test_df['YIELD']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_model.fit(X_train_scaled, y_train)

y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Test RMSE without PCA: {test_rmse}, Test R^2: {test_r2}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, label='Test Data', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield (Without PCA)')
plt.legend()
plt.show()

# Find the optimal number of PCA components
holder_best = {}

for i in range(1, 101):
    pca = PCA(n_components=i)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    best_model.fit(X_train_pca, y_train)
    
    y_train_pred = best_model.predict(X_train_pca)
    y_test_pred = best_model.predict(X_test_pca)
    
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f'Test RMSE with PCA components {i}: {test_rmse}, Test R^2: {test_r2}')
    holder_best[i] = test_r2

max_key = max(holder_best, key=holder_best.get)
max_value = holder_best[max_key]
print(f"Best number of PCA components: {max_key} with R^2: {max_value}")

# Apply PCA to the scaled data with the best number of components
pca = PCA(n_components=max_key)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

best_model.fit(X_train_pca, y_train)

y_train_pred = best_model.predict(X_train_pca)
y_test_pred = best_model.predict(X_test_pca)

test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Test RMSE with optimal PCA components: {test_rmse}, Test R^2: {test_r2}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, label='Test Data', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield (With PCA)')
plt.legend()
plt.show()










