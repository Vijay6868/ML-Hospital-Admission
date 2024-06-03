import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# Load the data
file_path = '/Users/vj/Documents/Data/ML-Hospital-Admission/Dataset/HDHI Admission data.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Convert 'D.O.A' to datetime
df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')

# Drop rows with missing 'D.O.A' values
df = df.dropna(subset=['D.O.A'])

# Filter data from mid-2018
start_date = pd.to_datetime('2018-06-01')
df = df[df['D.O.A'] <= start_date]

# Aggregate data by day and gender
daily_gender_counts = df.groupby(['D.O.A', 'GENDER']).size().unstack(fill_value=0)

# Reset index to have 'D.O.A' as a column
daily_gender_counts.reset_index(inplace=True)
daily_gender_counts.columns = ['ds', 'Female', 'Male']

# Create additional features
daily_gender_counts['day_of_week'] = daily_gender_counts['ds'].dt.dayofweek
daily_gender_counts['month'] = daily_gender_counts['ds'].dt.month
daily_gender_counts['day_of_year'] = daily_gender_counts['ds'].dt.dayofyear
daily_gender_counts['year'] = daily_gender_counts['ds'].dt.year

# Set the target variable (here we use Female count as the target for simplicity)
X = daily_gender_counts.drop(['ds', 'Female', 'Male'], axis=1)
y = daily_gender_counts['Female'] > daily_gender_counts['Male']  # Binary target: True if more females than males, else False

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = XGBClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot feature importances
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Create a DataFrame for actual and predicted values
results = X_test.copy()
results['Actual'] = y_test
results['Predicted'] = y_pred

# Merge with the original DataFrame to get the dates
results = results.merge(daily_gender_counts[['ds']], left_index=True, right_index=True)

# Sort by date
results = results.sort_values(by='ds')

# Plot actual vs. predicted admissions over time
plt.figure(figsize=(14, 8))
plt.plot(results['ds'], results['Actual'], marker='o', linestyle='-', label='Actual')
plt.plot(results['ds'], results['Predicted'], marker='x', linestyle='--', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Female > Male Admissions')
plt.title('Actual vs. Predicted Gender Distribution Over Time')
plt.legend()
plt.show()
