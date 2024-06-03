import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# Load the data
file_path = '/Users/vj/Documents/Data/ML-Hospital-Admission/Dataset/HDHI Admission data.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Convert 'D.O.A' to datetime
def parse_date(date):
    for fmt in ("%m/%d/%Y", "%d/%m/%Y"):
        try:
            return pd.to_datetime(date, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(date, format='ISO8601', errors='coerce')

df['D.O.A'] = df['D.O.A'].apply(parse_date)
df = df.dropna(subset=['D.O.A'])  # Drop rows where 'D.O.A' could not be parsed

# Filter data from mid-2018
start_date = pd.to_datetime('2018-06-01')
df = df[df['D.O.A'] <= start_date]

# Group by date and count admissions
admissions_per_day = df.groupby('D.O.A').size().reset_index(name='admissions')
admissions_per_day.rename(columns={'D.O.A': 'ds', 'admissions': 'y'}, inplace=True)

# Prepare the data
data = admissions_per_day['y'].values

# Create features for XGBoost
def create_features(data, lags):
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data[i-lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

lags = 30  # Number of days to look back
X, y = create_features(data, lags)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(X_train, y_train)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Calculate performance metrics
train_mae = mean_absolute_error(y_train, train_predict)
train_mse = mean_squared_error(y_train, train_predict)
test_mae = mean_absolute_error(y_test, test_predict)
test_mse = mean_squared_error(y_test, test_predict)

print(f"Train Mean Absolute Error: {train_mae}")
print(f"Train Mean Squared Error: {train_mse}")
print(f"Test Mean Absolute Error: {test_mae}")
print(f"Test Mean Squared Error: {test_mse}")

import numpy as np

# Calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate MAPE for the test set
test_mape = mean_absolute_percentage_error(y_test, test_predict)
test_accuracy = 100 - test_mape

print(f"Test Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Calculate MAPE for the test set
test_mape = mean_absolute_percentage_error(y_test, test_predict)
test_accuracy = 100 - test_mape

print(f"Test Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(admissions_per_day['ds'], data, label='Actual', marker='o', linestyle='-')
plt.plot(admissions_per_day['ds'][lags:train_size+lags], train_predict, label='Train Predictions', linestyle='--')
plt.plot(admissions_per_day['ds'][train_size+lags:], test_predict, label='Test Predictions', linestyle='--')
plt.title('Actual vs Predicted Admissions')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.legend()
plt.grid(True)
plt.show()


