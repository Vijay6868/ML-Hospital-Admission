import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import p

# Load your preprocessed data
# train_features, train_target, test_features, test_target = your_data_loading_function()

# Convert the dataset into an optimized data structure called Dmatrix
dtrain = xgb.DMatrix(train_features, label=train_target)
dtest = xgb.DMatrix(test_features)

# Set the parameters for XGBoost
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Train the model
num_round = 100
bst = xgb.train(params, dtrain, num_round)

# Make predictions
predictions = bst.predict(dtest)

# Calculate RMSE
rmse = mean_squared_error(test_target, predictions, squared=False)
print(f"RMSE: {rmse}")

# Plot the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_target, label='Actual', marker='o')
plt.plot(predictions, label='Predicted', marker='x')
plt.title('Actual vs Predicted Values')
plt.xlabel('Observations')
plt.ylabel('Targets')
plt.legend()
plt.show()
