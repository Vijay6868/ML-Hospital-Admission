import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = '/Users/vj/Documents/Data/ML-Hospital-Admission/Dataset/HDHI Admission data.csv'
df = pd.read_csv(file_path)

# Strip any leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Check if 'D.O.A' exists and if it doesn't, identify potential name issues
if 'D.O.A' not in df.columns:
    raise KeyError("Column 'D.O.A' not found in the dataset")

# Function to parse dates with mixed formats
def date_parse(str_date):
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y'):
        try:
            return pd.to_datetime(str_date, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"No valid date format found for {str_date}")

# Apply the function to the 'D.O.A' column

df['D.O.A'] = df['D.O.A'].apply(date_parse)

start_date = pd.to_datetime('2018-06-01')
df = df[df['D.O.A'] <= start_date]

# Aggregate the data to get the number of admissions per day
admissions_per_day = df.groupby('D.O.A').size().reset_index(name='admissions')

# Prepare the data for Random Forest Regressor
admissions_per_day.rename(columns={'D.O.A': 'date', 'admissions': 'admissions'}, inplace=True)

# Split the data into features (X) and target (y)
X = admissions_per_day[['date']].copy()
y = admissions_per_day['admissions'].copy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(X_test['date'], y_test, label='Actual', marker='o', linestyle='-')
plt.plot(X_test['date'], y_pred, label='Predicted', marker='x', linestyle='--')
plt.title('Actual vs Predicted Admissions')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.legend()
plt.grid(True)
plt.show()
