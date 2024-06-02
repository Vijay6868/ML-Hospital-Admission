import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA

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
    for fmt in ('%m/%d/%Y', '%d/%m/%Y'):
        try:
            return pd.to_datetime(str_date, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"No valid date format found for {str_date}")

# Apply the function to the 'D.O.A' column
df['D.O.A'] = df['D.O.A'].apply(date_parse)

# Set 'D.O.A' as the index
df.set_index('D.O.A', inplace=True)

# Aggregate the data to get the number of admissions per day
admissions_per_day = df.groupby(df.index).size().reset_index(name='admissions')

# Ensure the index is datetime
admissions_per_day.set_index('D.O.A', inplace=True)
admissions_per_day.index = pd.to_datetime(admissions_per_day.index)

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(admissions_per_day.index, admissions_per_day['admissions'], marker='o', linestyle='-')
plt.title('Daily Hospital Admissions')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.grid(True)
plt.show()

# Split the data into training and testing sets
train_size = int(len(admissions_per_day) * 0.8)
train, test = admissions_per_day[:train_size], admissions_per_day[train_size:]

# Use auto_arima to find the best parameters
stepwise_model = auto_arima(train['admissions'], start_p=1, start_q=1,
                            max_p=5, max_q=5, seasonal=False,
                            trace=True, error_action='ignore', suppress_warnings=True)
print(stepwise_model.summary())

# Fit ARIMA model
model = ARIMA(train['admissions'], order=stepwise_model.order)
fitted_model = model.fit()

# Make predictions
predictions = fitted_model.forecast(steps=len(test))

# Evaluate the model
mse = mean_squared_error(test['admissions'], predictions)
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['admissions'], label='Actual', marker='o', linestyle='-')
plt.plot(test.index, predictions, label='Predicted', marker='x', linestyle='--')
plt.title('Actual vs Predicted Admissions')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.legend()
plt.grid(True)
plt.show()
