import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

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

# Aggregate the data by day to get total duration of stay per day
daily_duration = df.groupby('D.O.A')['DURATION OF STAY'].sum().reset_index()

# Create additional features
daily_duration['day_of_week'] = daily_duration['D.O.A'].dt.dayofweek
daily_duration['month'] = daily_duration['D.O.A'].dt.month
daily_duration['day_of_year'] = daily_duration['D.O.A'].dt.dayofyear
daily_duration['year'] = daily_duration['D.O.A'].dt.year

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Hospital Admission Dashboard"),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=daily_duration['D.O.A'].min(),
        end_date=daily_duration['D.O.A'].max(),
        display_format='YYYY-MM-DD'
    ),
    dcc.Graph(id='total-duration-graph')
])

@app.callback(
    Output('total-duration-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graph(start_date, end_date):
    filtered_df = daily_duration[(daily_duration['D.O.A'] >= start_date) & (daily_duration['D.O.A'] <= end_date)]
    fig = px.line(filtered_df, x='D.O.A', y='DURATION OF STAY', title='Total Duration of Stay Over Time')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
