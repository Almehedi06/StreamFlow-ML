# %%
# Data manipulation and numerical operations
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations

# Visualization libraries
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns            # For statistical data visualization

# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose  # For decomposing time series data
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # For plotting autocorrelation functions
from scipy.stats import skew, kurtosis  # For statistical analysis (skewness and kurtosis)

# Machine learning libraries
from sklearn.ensemble import RandomForestRegressor  # Random Forest regression model
from sklearn.impute import SimpleImputer  # For handling missing data
from sklearn.model_selection import train_test_split  # For splitting datasets into training and testing sets
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For feature scaling and encoding categorical variables
from sklearn.compose import ColumnTransformer  # For applying transformations to specific columns
from sklearn.pipeline import Pipeline  # For creating a machine learning pipeline

# Set visualization style
# %matplotlib inline  # Enable inline plotting in Jupyter notebooks
sns.set(color_codes=True)  # Set the seaborn color codes for better aesthetics

# Google Colab specific
from google.colab import drive  # For mounting Google Drive in Colab
drive.mount('/content/drive')  # Mount Google Drive to access files

# %%
df = pd.read_csv('/content/drive/My Drive/CAMEL_ML_internal/camel_ml_df.csv')

# %% [markdown]
# #EDA

# %%
print(list(df))

# %%


# %%
# Convert date column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# %%
gauge_stations = [1415000, 1414500, 1434025, 1350140, 1510000]

# Subset the dataframe for the specified gauge stations
subset_df = df[df['gauge_id'].isin(gauge_stations)]

# %%
df = subset_df.copy()

# %%
df

# %%
# Create a dictionary to store the dataframes
dataframes = {gauge_id: df[df['gauge_id'] == gauge_id] for gauge_id in gauge_stations}

# Export each dataframe as CSV and display them
for gauge_id, dataframe in dataframes.items():
    csv_data = dataframe.to_csv(index=False)
    print(f"Gauged Station: {gauge_id}")
    print(csv_data)

# %%
df[['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp', 'streamflow']].describe()

# %%
# Check for missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print(missing_percentage)

# %% [markdown]
# Time Series Visualization

# %%
# Time Series Visualization
df[['prcp', 'srad', 'tmax', 'tmin', 'vp', 'streamflow']].plot(subplots=True, figsize=(15, 10))
plt.show()

# %%
# Aggregate data by month and plot
monthly_df = df.resample('M').mean()
monthly_df[['prcp', 'srad', 'tmax', 'tmin', 'vp', 'streamflow']].plot(subplots=True, figsize=(15, 10))
plt.show()

# %%
# Aggregate data by yearly and plot
yearlyly_df = df.resample('Y').mean()
yearlyly_df[['prcp', 'srad', 'tmax', 'tmin', 'vp', 'streamflow']].plot(subplots=True, figsize=(15, 10))
plt.show()

# %%
# Plot the time series of streamflow
plt.plot(df['streamflow'], label='Streamflow')
plt.xlabel('Date')
plt.ylabel('Streamflow')
plt.title('Time Series of Streamflow')
plt.legend()
plt.show()

# %%
plt.plot(df['prcp'], label='Precipitation', color='orange')
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.title('Time Series of Precipitation')
plt.legend()
plt.show()

# %% [markdown]
# Statistical Analysis

# %%
df[['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp', 'streamflow']].corr()

# %% [markdown]
# Prompt: I have a multivariate time series dataset, df with one target and multiple inputs. The data are collected at different location and have site IDs. At each site, there are mutiple observations of the variables. Now, give me a plot showing the correlation of each input with the output for each storm. Consider inputs 'dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp' and target 'streamflow'. Site ID is 'gauge_id'.    

# %%
input_vars = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp', 'streamflow']

# Calculate the correlation matrix
corr_matrix = df[input_vars].corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# %%
# Define the specific variables for plotting
vars_to_plot = ['streamflow', 'prcp', 'tmax']

# Plot streamflow vs prcp
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='prcp', y='streamflow')
plt.title('Streamflow vs Precipitation')
plt.xlabel('Precipitation (prcp)')
plt.ylabel('Streamflow')
plt.show()

# Plot streamflow vs tmax
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='tmax', y='streamflow')
plt.title('Streamflow vs Temperature (tmax)')
plt.xlabel('Maximum Temperature (tmax)')
plt.ylabel('Streamflow')
plt.show()

# %%
# Define the specific variables for plotting
variables = ['prcp', 'tmax', 'dayl', 'srad', 'tmin', 'vp']

# Create scatter plots
for var in variables:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=var, y='streamflow')
    plt.title(f'Streamflow vs {var}')
    plt.xlabel(var)
    plt.ylabel('Streamflow')
    plt.show()

# %% [markdown]
# Distribution Analysis

# %%
df_1 = df[['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp', 'streamflow']]

# %%
# Distribution Analysis
for column in df_1.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df_1[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

    print(f'Skewness of {column}: {skew(df_1[column])}')
    print(f'Kurtosis of {column}: {kurtosis(df_1[column])}')

# %%
#Display first ten unique values of a column.
unique_values = df['gauge_id'].unique()[:10]
print(unique_values)

# %%
number_of_unique_gauge_ids = df['gauge_id'].nunique()

print(f"The number of unique gauge IDs is: {number_of_unique_gauge_ids}")

# %%
# Specific values to subset
#values_to_subset = [1013500, 1022500, 1030500, 1031500, 1047000, 1052500, 1054200, 1055000, 1057000, 1073000]
values_to_subset = [1013500]

# Subset DataFrame
subset_df = df[df['gauge_id'].isin(values_to_subset)]

# %%
subset_df['streamflow'].hist()

# %%
#How can I check if the time series has regular interval?
subset_df

# %%
df = subset_df.copy()

# %%
df['gauge_id'].nunique()

# %%
df['date'] = pd.to_datetime(df['date'])

# %%
df = df.set_index('date')

# %% [markdown]
# ##Map

# %% [markdown]
# Prompt: I have a dataset that consists of a multivariate time series of hydrologic variables, which is depicted in the attached picture. The dataset includes multiple gauge stations identified by their Gauge IDs, and each station has multiple observations of various variables. Could you please provide a code to calculate the median streamflow for each unique gauge station? In the end, I would like to have a dataset containing all unique gauge stations along with the median values of their respective variables.

# %%
import pandas as pd

# Group by 'gauge_id' and calculate the median for each group
median_df = df.groupby('gauge_id').median().reset_index()
median_df

max_df = df.groupby('gauge_id').max().reset_index()
max_df

# %%
max_df = max_df.dropna()

# %%
import folium
import geopandas as gpd
import branca.colormap as cm

# Assuming max_df is already defined
# Create a GeoDataFrame
gmax_df = gpd.GeoDataFrame(max_df, geometry=gpd.points_from_xy(max_df['lon_cen'], max_df['lat_cen']))

# Create a colormap for streamflow values
colormap = cm.linear.YlGnBu_09.scale(gmax_df['streamflow'].min(), gmax_df['streamflow'].max())
colormap.caption = 'Streamflow'

# Create a map centered around the mean latitude and longitude
m = folium.Map(location=[gmax_df['lat_cen'].mean(), gmax_df['lon_cen'].mean()], zoom_start=5)

# Add points to the map with CircleMarker to visualize streamflow
for idx, row in gmax_df.iterrows():
    folium.CircleMarker(
        location=[row['lat_cen'], row['lon_cen']],
        radius=10,  # Increased radius for better visualization
        popup=f"Gauge ID: {row['gauge_id']}<br>Streamflow: {row['streamflow']}",
        color=colormap(row['streamflow']),  # Use color gradient based on streamflow
        fill=True,
        fill_color=colormap(row['streamflow']),
        fill_opacity=0.7
    ).add_to(m)

# Add the colormap to the map
colormap.add_to(m)

# Save the map
m.save('map.html')

# Display the map in Jupyter Notebook (if needed)
m

# %% [markdown]
# #New input features

# %%
df = df.rename(columns={'Qmm': 'Runoff'})

# %%
#df['Runoff'] = pd.to_numeric(df['Runoff'],errors = 'coerce')

# %%
from scipy import stats
# Add a small constant to avoid zero values
df['logP'] = df['P'] + 1e-5

# # Perform the Box-Cox transformation
# df['P'], lambda_ = stats.boxcox(df['P'])           # To display the botton 5 rows

# %%
df['logP'] = np.log(df['P'])

# %%
df['logRunoff'] = np.log(df['Runoff'])

# %%
start_date = '2000-01-01 00:00'
num_hours = len(df)  # Assuming df is your DataFrame containing rainfall and discharge columns

# Create a DateTimeIndex with hourly frequency
timestamps = pd.date_range(start=start_date, periods=num_hours, freq='H')

# Assign the timestamps as a new column in the DataFrame
df['timestamp'] = timestamps

# %%
df

# %%
# Assuming your date and time column is named 'timestamp'
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract the components into separate columns
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

# Drop the original 'timestamp' column if desired
df = df.drop('timestamp', axis=1)

# %%
# Create the binary feature
df['rain_binary'] = np.where(df['P'] > 0, 1, 0)

# %%
# Create a new column for accumulated P after each week
df['accumulated_P_daily'] = df['P'].rolling(window=24).sum()

# Fill NaN values with 0
df['accumulated_P_daily'].fillna(0, inplace=True)

# %%
# Create a new column for accumulated P after each week
df['accumulated_P_weekly'] = df['P'].rolling(window=7*24).sum()

# Fill NaN values with 0
df['accumulated_P_weekly'].fillna(0, inplace=True)

# %%
# Initialize a new column to store the accumulated sum of 'P'
df['AP'] = 0

# Initialize variables to keep track of the accumulated sum
accumulated_sum = 0
is_rain_event = False

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    if row['rain_binary'] == 1:
        accumulated_sum += row['P']
        is_rain_event = True
    elif row['rain_binary'] == 0 and is_rain_event:
        df.at[index, 'AP'] = accumulated_sum
        accumulated_sum = 0
        is_rain_event = False

# %%
print(df.shape)

# %%
df.describe()

# %% [markdown]
# 
# 
# ---
# 
# 

# %%
print(list(df))

# %%
df = df.dropna()

# %%
df.isna().sum()

# %% [markdown]
# #Feature Enginerring

# %% [markdown]
# #RF

# %% [markdown]
# ##All gauges

# %%
# Get unique gauge_ids
unique_gauge_ids = df['gauge_id'].unique()

# Define the number of samples for each set
num_train = 470
num_dev = 68
num_test = 136

# Randomly sample unique gauge_ids for train, dev, and test sets
train_gauge_ids = np.random.choice(unique_gauge_ids, num_train, replace=False)
remaining_gauge_ids = np.setdiff1d(unique_gauge_ids, train_gauge_ids)
dev_gauge_ids = np.random.choice(remaining_gauge_ids, num_dev, replace=False)
test_gauge_ids = np.setdiff1d(remaining_gauge_ids, dev_gauge_ids)

# Create the subsets
train_df = df[df['gauge_id'].isin(train_gauge_ids)]
dev_df = df[df['gauge_id'].isin(dev_gauge_ids)]
test_df = df[df['gauge_id'].isin(test_gauge_ids)]

# Feature sets
features = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
features_reduced = ['prcp', 'tmax']

# Target variable
target = ['streamflow']

# Full feature set
X_train = train_df[features]
y_train = train_df[target]

X_val = dev_df[features]
y_val = dev_df[target]

X_test = test_df[features]
y_test = test_df[target]

# Reduced feature set
X_train_reduced = train_df[features_reduced]
y_train_reduced = train_df[target]

X_val_reduced = dev_df[features_reduced]
y_val_reduced = dev_df[target]

X_test_reduced = test_df[features_reduced]
y_test_reduced = test_df[target]

# Feature scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)


# Define the Random Forest regressor
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_split=20,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=10,
    bootstrap=True
)

#Train the model
model.fit(X_train_scaled, y_train.values.ravel())

# %%
# Predict on the train and test sets
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Check predictions
print("Train predictions shape:", y_train_pred.shape)
print("Test predictions shape:", y_test_pred.shape)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}")

# %%
# Plot the observed vs predicted values for the training data
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_train)), y_train, label='Observed Train', color='b')
plt.plot(range(len(y_train)), y_train_pred, label='Predicted Train', color='c', linestyle='--')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Streamflow')
plt.title('Observed vs Predicted Streamflow (Training Data)')
plt.show()

# Plot the observed vs predicted values for the test data
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label='Observed Test', color='r')
plt.plot(range(len(y_test)), y_test_pred, label='Predicted Test', color='m', linestyle='--')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Streamflow')
plt.title('Observed vs Predicted Streamflow (Test Data)')
plt.show()

# %% [markdown]
# ##Each site sperately

# %% [markdown]
# Train/test scores for individual sites:
# Promt: I would like to get scores for the train and test for individual sites separately. Then plot the scores (R2).

# %%
# Define features and target
features = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
target = ['streamflow']

# Initialize the StandardScaler
scaler_X = StandardScaler()

# Fit the scaler on the entire training data
scaler_X.fit(train_df[features])

# Create the Linear Regression model
model = LinearRegression()

# Fit the model on the entire training data
model.fit(scaler_X.transform(train_df[features]), train_df[target].values.ravel())

# Function to perform feature engineering, prediction, and scoring for each site
def process_site(gauge_id, df, model, features, target, scaler):
    X_site = df[df['gauge_id'] == gauge_id][features]
    y_site = df[df['gauge_id'] == gauge_id][target]
    X_site_scaled = scaler.transform(X_site)
    y_site_pred = model.predict(X_site_scaled)
    r2 = r2_score(y_site, y_site_pred)
    return r2

# %%
# Initialize dictionaries to store R² scores for each site
train_r2_scores = {}
test_r2_scores = {}

# Get the unique gauge_ids in the train and test sets
train_gauge_ids = train_df['gauge_id'].unique()
test_gauge_ids = test_df['gauge_id'].unique()

# Process each site in the training set
for gauge_id in train_gauge_ids:
    train_r2_scores[gauge_id] = process_site(gauge_id, train_df, model, features, target, scaler_X)

# Process each site in the test set
for gauge_id in test_gauge_ids:
    test_r2_scores[gauge_id] = process_site(gauge_id, test_df, model, features, target, scaler_X)

# %%
# Convert the scores to DataFrames for easy reporting
train_r2_df = pd.DataFrame(list(train_r2_scores.items()), columns=['gauge_id', 'train_r2'])
test_r2_df = pd.DataFrame(list(test_r2_scores.items()), columns=['gauge_id', 'test_r2'])

# Print the resulting DataFrames
print("Training R² scores:")
print(train_r2_df)
print("\nTest R² scores:")
print(test_r2_df)

# %%
train_r2_df = train_r2_df.sort_values(by='train_r2', ascending=False)
test_r2_df = test_r2_df.sort_values(by='test_r2', ascending=False)

top_20_train = train_r2_df.head(20)
top_20_test = test_r2_df.head(20)

# %%
# Plot the top 20 R² scores for the training set
plt.figure(figsize=(14, 7))
plt.bar(top_20_train['gauge_id'].astype(str), top_20_train['train_r2'], alpha=0.6, label='Train R²')
plt.xlabel('Gauge ID')
plt.ylabel('Train R² Score')
plt.title('Top 20 Train R² Scores for Individual Gauge Stations')
plt.xticks(rotation=90)
plt.legend()
plt.show()

# Plot the top 20 R² scores for the test set
plt.figure(figsize=(14, 7))
plt.bar(top_20_test['gauge_id'].astype(str), top_20_test['test_r2'], alpha=0.6, label='Test R²')
plt.xlabel('Gauge ID')
plt.ylabel('Test R² Score')
plt.title('Top 20 Test R² Scores for Individual Gauge Stations')
plt.xticks(rotation=90)
plt.legend()
plt.show()

# %% [markdown]
# Strength in each site: Train/test separately within each site. Firt 80% train and rest 20% test.
# 

# %% [markdown]
# Overfitting/underfitting:
# Train score and test scores difference

# %% [markdown]
# Train with ten staions

# %%
# List of gauge IDs to process
gauge_ids = [1144000, 2143040, 2177000, 3504000, 6632400, 6903400, 7362100, 10234500, 14185900, 14222500]

# Define the Random Forest regressor
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_split=20,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=10,
    bootstrap=True
)

# Function to train and evaluate the model for a given gauge_id
def train_and_evaluate_rf(gauge_id, df):
    data = df[df['gauge_id'] == gauge_id].copy()

    # Check if there are enough data points
    if len(data) < 2:
        print(f"Not enough data for gauge ID: {gauge_id}")
        return

    # Prepare the features and target
    X = data[['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']]
    y = data['streamflow']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate scores
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the scores
    print(f"Gauge ID: {gauge_id}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print()

# Process each gauge ID
for gauge_id in gauge_ids:
    train_and_evaluate_rf(gauge_id, df)

# %%
# List of gauge IDs to process
gauge_ids = [1144000, 2143040, 2177000, 3504000, 6632400, 6903400, 7362100, 10234500, 14185900, 14222500]

# Define the Random Forest regressor
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_split=20,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=10,
    bootstrap=True
)

# Function to train and evaluate the model for a given gauge_id
def train_and_evaluate_rf(gauge_id, df):
    data = df[df['gauge_id'] == gauge_id].copy()

    # Check if there are enough data points
    if len(data) < 2:
        print(f"Not enough data for gauge ID: {gauge_id}")
        return

    # Prepare the features and target
    X = data[['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']]
    y = data['streamflow']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate scores
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the scores
    print(f"Gauge ID: {gauge_id}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Observed Discharge')
    plt.plot(y_pred, label='Predicted Discharge')
    plt.xlabel('Sample')
    plt.ylabel('Discharge')
    plt.title(f'Gauge ID: {gauge_id} - Random Forest Model')
    plt.legend()
    plt.show()

# Process each gauge ID
for gauge_id in gauge_ids:
    train_and_evaluate_rf(gauge_id, df)

# %% [markdown]
# Test scores for new unseen five stations

# %% [markdown]
# ##Train/test within sites

# %%
# In the following code, modify it to also make a dataset with the gauge_id, datetime, actual and predicted streamflow. All
# other existing things should stay in the code.

# <# ✅ Function to calculate NSE and PBIAS
# def calculate_nse(y_true, y_pred):
#     return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

# def calculate_pbias(y_true, y_pred):
#     return 100 * np.sum(y_true - y_pred) / np.sum(y_true)

# # ✅ Function to process a batch of gauge stations using Random Forest
# def process_gauges_individually_RF(df, gauge_ids_batch):
#     metrics = []

#     for gauge_id in gauge_ids_batch:
#         # Filter data for the current site
#         site_data = df[df['gauge_id'] == gauge_id]

#         # Split into 80% training and 20% testing
#         split_index = int(len(site_data) * 0.8)
#         train_site_data = site_data.iloc[:split_index]
#         test_site_data = site_data.iloc[split_index:]

#         # Initialize StandardScaler and Random Forest Regressor
#         scaler_X = StandardScaler()
#         model = RandomForestRegressor(
#             n_estimators=50,         # Increased trees for better stability
#             min_samples_split=5,
#             min_samples_leaf=2,
#             max_features='sqrt',
#             max_depth=5,             # Slightly deeper trees for balance
#             bootstrap=True,
#             random_state=42
#         )

#         # Scale the features based on the training data
#         X_train_scaled = scaler_X.fit_transform(train_site_data[features])
#         X_test_scaled = scaler_X.transform(test_site_data[features])

#         # Extract target variable
#         y_train = train_site_data[target].values.ravel()
#         y_test = test_site_data[target].values.ravel()

#         # Fit the model on the training data
#         model.fit(X_train_scaled, y_train)

#         # Predict on training and testing data
#         y_train_pred = model.predict(X_train_scaled)
#         y_test_pred = model.predict(X_test_scaled)

#         # Compute evaluation metrics
#         train_r2 = r2_score(y_train, y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
#         train_mse = mean_squared_error(y_train, y_train_pred)
#         test_mse = mean_squared_error(y_test, y_test_pred)
#         train_rmse = np.sqrt(train_mse)
#         test_rmse = np.sqrt(test_mse)
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
#         train_nse = calculate_nse(y_train, y_train_pred)
#         test_nse = calculate_nse(y_test, y_test_pred)
#         train_pbias = calculate_pbias(y_train, y_train_pred)
#         test_pbias = calculate_pbias(y_test, y_test_pred)

#         # Store the results
#         metrics.append({
#             'gauge_id': gauge_id,
#             'train_r2': train_r2, 'test_r2': test_r2,
#             'train_rmse': train_rmse, 'test_rmse': test_rmse,
#             'train_mse': train_mse, 'test_mse': test_mse,
#             'train_mae': train_mae, 'test_mae': test_mae,
#             'train_nse': train_nse, 'test_nse': test_nse,
#             'train_pbias': train_pbias, 'test_pbias': test_pbias
#         })

#         print(f"Gauge ID: {gauge_id} processed successfully.")

#     return pd.DataFrame(metrics)

# # ✅ Function to process gauge stations in chunks
# def process_gauges_in_chunks_RF(df, start_idx=0, chunk_size=30, results_file='RF_results.csv'):
#     try:
#         results_df = pd.read_csv(results_file)
#     except FileNotFoundError:
#         results_df = pd.DataFrame(columns=['gauge_id', 'train_r2', 'test_r2', 'train_rmse', 'test_rmse',
#                                            'train_mse', 'test_mse', 'train_mae', 'test_mae', 'train_nse',
#                                            'test_nse', 'train_pbias', 'test_pbias'])

#     unique_gauge_ids = df['gauge_id'].unique()
#     total_gauges = len(unique_gauge_ids)

#     gauge_ids_batch = unique_gauge_ids[start_idx:start_idx + chunk_size]
#     print(f"Processing gauge stations {start_idx + 1} to {min(start_idx + chunk_size, total_gauges)}...")

#     df_subset = df[df['gauge_id'].isin(gauge_ids_batch)]
#     metrics_df = process_gauges_individually_RF(df_subset, gauge_ids_batch)

#     results_df = pd.concat([results_df, metrics_df], ignore_index=True)
#     results_df.to_csv(results_file, index=False)

#     print("Results saved to", results_file)

# # ✅ Example usage:
# features = ['prcp', 'srad', 'tmax']  # Modified feature selection
# target = ['streamflow']

# process_gauges_in_chunks_RF(df, start_idx=49, chunk_size=674)>







import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_csv("/content/drive/My Drive/CAMEL_ML_internal/camel_ml_df.csv")
df['date'] = pd.to_datetime(df['date'])  # Ensure date column is in datetime format

# Function to calculate NSE and PBIAS
def calculate_nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

def calculate_pbias(y_true, y_pred):
    return 100 * np.sum(y_true - y_pred) / np.sum(y_true)

# Function to process a batch of gauge stations using Random Forest
def process_gauges_individually_RF(df, gauge_ids_batch):
    metrics = []
    predictions = []  # Store predictions with gauge_id and date

    for gauge_id in gauge_ids_batch:
        # Filter data for the current site
        site_data = df[df['gauge_id'] == gauge_id]

        # Split into 80% training and 20% testing
        split_index = int(len(site_data) * 0.8)
        train_site_data = site_data.iloc[:split_index]
        test_site_data = site_data.iloc[split_index:]

        # Initialize StandardScaler and Random Forest Regressor
        scaler_X = StandardScaler()
        model = RandomForestRegressor(
            n_estimators=50,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            max_depth=5,
            bootstrap=True,
            random_state=42
        )

        # Scale the features based on the training data
        X_train_scaled = scaler_X.fit_transform(train_site_data[features])
        X_test_scaled = scaler_X.transform(test_site_data[features])

        # Extract target variable
        y_train = train_site_data[target].values.ravel()
        y_test = test_site_data[target].values.ravel()

        # Fit the model on the training data
        model.fit(X_train_scaled, y_train)

        # Predict on training and testing data
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Compute evaluation metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_nse = calculate_nse(y_train, y_train_pred)
        test_nse = calculate_nse(y_test, y_test_pred)
        train_pbias = calculate_pbias(y_train, y_train_pred)
        test_pbias = calculate_pbias(y_test, y_test_pred)

        # Store the results
        metrics.append({
            'gauge_id': gauge_id,
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_nse': train_nse, 'test_nse': test_nse,
            'train_pbias': train_pbias, 'test_pbias': test_pbias
        })

        # Store actual vs predicted streamflow for train and test sets
        train_results = pd.DataFrame({
            'gauge_id': gauge_id,
            'date': train_site_data['date'],
            'actual_streamflow': y_train,
            'predicted_streamflow': y_train_pred
        })

        test_results = pd.DataFrame({
            'gauge_id': gauge_id,
            'date': test_site_data['date'],
            'actual_streamflow': y_test,
            'predicted_streamflow': y_test_pred
        })

        predictions.append(train_results)
        predictions.append(test_results)

        print(f"Gauge ID: {gauge_id} processed successfully.")

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)

# Function to process gauge stations in chunks and save results with dynamic filenames
def process_gauges_in_chunks_RF(df, start_idx=0, chunk_size=30, results_dir='/content/drive/My Drive/CAMEL_ML_internal/'):
    unique_gauge_ids = df['gauge_id'].unique()
    total_gauges = len(unique_gauge_ids)

    # Determine the range of gauge IDs for the filenames
    start_gauge_id = unique_gauge_ids[start_idx]
    end_gauge_id = unique_gauge_ids[min(start_idx + chunk_size - 1, total_gauges - 1)]

    # Define file names dynamically based on gauge ID range
    results_file = os.path.join(results_dir, f'RF_scores_{start_gauge_id}_{end_gauge_id}.csv')
    predictions_file = os.path.join(results_dir, f'RF_predictions_{start_gauge_id}_{end_gauge_id}.csv')

    # Try loading existing results if they exist
    try:
        results_df = pd.read_csv(results_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=['gauge_id', 'train_r2', 'test_r2', 'train_rmse', 'test_rmse',
                                           'train_mse', 'test_mse', 'train_mae', 'test_mae', 'train_nse',
                                           'test_nse', 'train_pbias', 'test_pbias'])

    try:
        predictions_df = pd.read_csv(predictions_file)
    except FileNotFoundError:
        predictions_df = pd.DataFrame(columns=['gauge_id', 'date', 'actual_streamflow', 'predicted_streamflow'])

    # Select gauge stations for processing
    gauge_ids_batch = unique_gauge_ids[start_idx:start_idx + chunk_size]
    print(f"Processing gauge stations {start_idx + 1} to {min(start_idx + chunk_size, total_gauges)}...")

    df_subset = df[df['gauge_id'].isin(gauge_ids_batch)]
    metrics_df, pred_df = process_gauges_individually_RF(df_subset, gauge_ids_batch)

    # Append results and save
    results_df = pd.concat([results_df, metrics_df], ignore_index=True)
    predictions_df = pd.concat([predictions_df, pred_df], ignore_index=True)

    results_df.to_csv(results_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)

    print(f"Results saved to {results_file}")
    print(f"Predictions saved to {predictions_file}")

# Example usage
features = ['dayl', 'prcp', 'srad', 'tmax', 'tmin', 'vp']
target = ['streamflow']

process_gauges_in_chunks_RF(df, start_idx=0, chunk_size=100)

# %% [markdown]
# Prompt: Generate a point plot to compare the train and test R² scores across sites. Ensure that the x-axis does not display gauge IDs and that there are no connecting lines between the points. Limit the y-axis to a range of -1 to 1.

# %%
# Merging the train and test R² scores into a single DataFrame for plotting
r2_scores_df = pd.merge(train_r2_df, test_r2_df, on='gauge_id')

# Melt the DataFrame to have a long-form structure suitable for seaborn's pointplot
r2_scores_long = pd.melt(r2_scores_df, id_vars='gauge_id', value_vars=['train_r2', 'test_r2'],
                         var_name='Dataset', value_name='R² Score')

# Set the figure size
plt.figure(figsize=(14, 8))

# Create a point plot without connecting lines
sns.pointplot(data=r2_scores_long, x='gauge_id', y='R² Score', hue='Dataset', markers=["o", "X"], linestyles="")

# Add labels and title
plt.xlabel('Gauge ID')
plt.ylabel('R² Score')
plt.title('Train and Test R² Scores by Site')

# Limit y-axis ticks
plt.ylim(-1, 1)

# Remove the x-axis gauge IDs for better readability
plt.xticks([])

# Show the plot
plt.tight_layout()
plt.show()

# %% [markdown]
# Prompt: Get the difference between the train and test scores for the sites and show plot.

# %%
# Merging the train and test R² scores into a single DataFrame for plotting
r2_scores_df = pd.merge(train_r2_df, test_r2_df, on='gauge_id')

# Calculate the difference between train and test R² scores for each site
r2_scores_df['r2_diff'] = r2_scores_df['train_r2'] - r2_scores_df['test_r2']

# Set the figure size
plt.figure(figsize=(14, 8))

# Create a bar plot of the R² score differences with a distinct color using Matplotlib
plt.bar(r2_scores_df['gauge_id'], r2_scores_df['r2_diff'], color='red')

# Add labels and title
plt.xlabel('Gauge ID')
plt.ylabel('Difference in R² Scores (Train - Test)')
plt.title('Difference Between Train and Test R² Scores by Site')

# Limit y-axis from -1 to 2
# plt.ylim(-1, 2)

# Remove the x-axis gauge IDs for better readability
plt.xticks([])

# Show the plot
plt.tight_layout()
plt.show()

# %% [markdown]
# Bias:Variance: Overfit and underfit

# %%
r2_scores_df = pd.merge(train_r2_df, test_r2_df, on='gauge_id')

# Calculate the difference between train and test R² scores for each site
r2_scores_df['r2_diff'] = r2_scores_df['train_r2'] - r2_scores_df['test_r2']

# Add a column for overfit/underfit classification
r2_scores_df['fit_status'] = r2_scores_df['r2_diff'].apply(lambda x: 'overfit' if x > 0 else 'underfit')

# Display the DataFrame with the new column
print(r2_scores_df)

# %%
# Count the occurrences of overfit and underfit
fit_counts = r2_scores_df['fit_status'].value_counts()

# Calculate the percentage of overfit and underfit
fit_percentages = r2_scores_df['fit_status'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single DataFrame
fit_summary = pd.DataFrame({
    'Count': fit_counts,
    'Percentage (%)': fit_percentages
})

# Display the summary
print(fit_summary)

# %%
r2_scores_df

# %% [markdown]
# #Spplementary

# %%
first_100_gauge_ids = df['gauge_id'].unique()[:5]

df_100 = df[df['gauge_id'].isin(first_100_gauge_ids)]

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create the Random Forest regressor
model2 = RandomForestRegressor(n_estimators = 300, min_samples_split = 20, min_samples_leaf = 2, max_features = 'sqrt', max_depth = 10, bootstrap = True)

#'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 300

# Train the model
model2.fit(X_train, y_train)

# %% [markdown]
# HP tuning

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Assuming df_100 is already defined and loaded with your data
# Extract features and target variable
features = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
target = 'streamflow'  # Assuming 'streamflow' is the target column

X = df_100[features]
y = df_100[target]

# Split the data into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create the Random Forest regressor
model = RandomForestRegressor()

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters from grid search
best_params = grid_search.best_params_

# Print the best parameters
print("Best parameters found by GridSearchCV:")
print(best_params)

# %% [markdown]
# Model performance on train set

# %%
# Make predictions on the train set
y_train_pred = model2.predict(X_train)

# %%
# Calculate the R2 score
r2_train = r2_score(y_train, y_train_pred)
print("R2 Score (train):", r2_train)

# %% [markdown]
# Model Performance on Test Set

# %%
# Make predictions on the test set
y_test_pred = model2.predict(X_test)

# %%
# Calculate the R2 score
r2_test = r2_score(y_test, y_test_pred)
print("R2 Score (test):", r2_test)

# %%
y_test.shape

# %%
# Compute R2 score for the train set
r2 = r2_score(df['Original Data'], df['Predicted Data'])
print('The R2 for the train data is:', r2)

# %%
A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)

# %%
B.shape

# %%
x = np.random.rand(4, 5)
y = np.sum(x, axis=1)

# %%
# Assume 'df' is your DataFrame and it has already been split into train, dev, and test sets

# Convert gauge_id to categorical
df['gauge_id'] = df['gauge_id'].astype('category')

# Feature sets
features = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
target = ['streamflow']

# Include 'gauge_id' as a feature for all data
X = df[features + ['gauge_id']]
y = df[target]

# Get all unique categories in the entire dataset for 'gauge_id'
all_gauge_ids = df['gauge_id'].unique()

# Define the ColumnTransformer with predefined categories
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features),
        ('cat', OneHotEncoder(categories=[all_gauge_ids], drop='first'), ['gauge_id'])
    ])

# Define the Random Forest regressor
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_split=20,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=10,
    bootstrap=True,
    random_state=42
)

# Create a pipeline that includes the preprocessor and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Split the data into training, validation, and test sets
# Assuming you already have train_df, dev_df, and test_df created based on your criteria
X_train = train_df[features + ['gauge_id']]
y_train = train_df[target]

X_val = dev_df[features + ['gauge_id']]
y_val = dev_df[target]

X_test = test_df[features + ['gauge_id']]
y_test = test_df[target]

# Train the model on the training set
pipeline.fit(X_train, y_train.values.ravel())

# Predict on the train and test sets
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Check predictions
print("Train predictions shape:", y_train_pred.shape)
print("Test predictions shape:", y_test_pred.shape)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}")

# %%



