import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/Users/vj/Documents/Data/ML-Hospital-Admission/Dataset/HDHI Admission data.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Convert 'D.O.A' to datetime
df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')

# Drop rows with missing 'D.O.A' values
df = df.dropna(subset=['D.O.A'])

# Filter data from mid-2018 if necessary
start_date = pd.to_datetime('2018-06-01')
df = df[df['D.O.A'] <= start_date]

# Ensure 'DURATION OF STAY' is a numeric column
df['DURATION OF STAY'] = pd.to_numeric(df['DURATION OF STAY'], errors='coerce')

# Drop rows with missing 'DURATION OF STAY' values
df = df.dropna(subset=['DURATION OF STAY'])

# Create additional features
df['day_of_week'] = df['D.O.A'].dt.dayofweek
df['month'] = df['D.O.A'].dt.month
df['day_of_year'] = df['D.O.A'].dt.dayofyear
df['year'] = df['D.O.A'].dt.year

# Define the features and target variable
X = df[['day_of_week', 'month', 'day_of_year', 'year']]
y = df['DURATION OF STAY']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = XGBRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Plot feature importances
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Plot actual vs. predicted duration of stay
plt.figure(figsize=(14, 8))
plt.plot(y_test.values, label='Actual', marker='o', linestyle='')
plt.plot(y_pred, label='Predicted', marker='x', linestyle='')
plt.xlabel('Sample')
plt.ylabel('Duration of Stay')
plt.title('Actual vs. Predicted Duration of Stay')
plt.legend()
plt.show()
