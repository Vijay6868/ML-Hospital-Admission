import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# Load CSV data
data = pd.read_csv("Dataset/HDHI Admission data.csv")

# Extract date of admission (D.O.A)
data["D.O.A"] = pd.to_datetime(data["D.O.A"], infer_datetime_format=True)


# Group data by month and count admissions
monthly_admissions = (
    data.groupby(data["D.O.A"].dt.month)
    .size()
    .to_frame(name="Admissions")
    .reset_index()
)

# Set index as the month
monthly_admissions.set_index("D.O.A", inplace=True)

# Perform time series decomposition
decomposition = seasonal_decompose(
    monthly_admissions["Admissions"], model="additive", period=12
)

# Extract trend, seasonality, and residuals
trend = decomposition.trend
seasonality = decomposition.seasonal
residuals = decomposition.resid

# Visualize the decomposed components
import matplotlib.pyplot as plt

plt.plot(monthly_admissions.index, trend, label="Trend")
plt.plot(monthly_admissions.index, seasonality, label="Seasonality")
plt.plot(monthly_admissions.index, residuals, label="Residuals")
plt.legend()
plt.show()