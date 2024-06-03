import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'Dataset/HDHI Admission data.csv'
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

# Filter the data from mid-2018 onwards
start_date = pd.to_datetime('2018-06-01')
df = df[df['D.O.A'] >= start_date]

# Debugging: Print the filtered data
print("Filtered Data:")
print(df.head())

# Aggregate the data to get the number of admissions per day
admissions_per_day = df.groupby('D.O.A').size().reset_index(name='admissions')

# Debugging: Print the admissions per day
print("Admissions Per Day:")
print(admissions_per_day.head())

# Prepare the data for Prophet
admissions_per_day.rename(columns={'D.O.A': 'ds', 'admissions': 'y'}, inplace=True)

# Debugging: Print the prepared data
print("Prepared Data for Prophet:")
print(admissions_per_day.head())

# Split the data into training and testing sets
train_size = int(len(admissions_per_day) * 0.8)
train, test = admissions_per_day[:train_size], admissions_per_day[train_size:]

# Debugging: Print the training and testing sets
print("Training Set:")
print(train.head())
print("Testing Set:")
print(test.head())

# Fit Prophet model
model = Prophet()
model.fit(train)

# Make predictions
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# Extract predictions
predictions = forecast[['ds', 'yhat']].iloc[-len(test):]

# Evaluate the model
mse = mean_squared_error(test['y'], predictions['yhat'])
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(test['ds'], test['y'], label='Actual', marker='o', linestyle='-')
plt.plot(predictions['ds'], predictions['yhat'], label='Predicted', marker='x', linestyle='--')
plt.title('Actual vs Predicted Admissions')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.legend()
plt.grid(True)
plt.show()
