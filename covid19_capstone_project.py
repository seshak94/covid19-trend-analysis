# COVID-19 Capstone Project: Trend Analysis and Forecasting

# Importing necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# Step 1: Load and Prepare the Dataset
# Loading the dataset
data = pd.read_csv('covid_19_clean_complete.csv')

# Converting Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Handling missing values in Province/State (replace empty with 'Unknown')
data['Province/State'] = data['Province/State'].fillna('Unknown')

# Step 2: Trend Analysis and Visualization

# Aggregating global data by date
global_data = data.groupby('Date').agg({
    'Confirmed': 'sum',
    'Recovered': 'sum',
    'Deaths': 'sum',
    'Active': 'sum'
}).reset_index()

# Computing daily new confirmed cases and recovery rate globally
global_data['Daily_New_Confirmed'] = global_data['Confirmed'].diff().fillna(0)
global_data['Daily_Recovered'] = global_data['Recovered'].diff().fillna(0)
global_data['Recovery_Rate'] = global_data['Recovered'] / global_data['Confirmed'].replace(0, np.nan)

# Filtering data for India
india_data = data[data['Country/Region'] == 'India'].groupby('Date').agg({
    'Confirmed': 'sum',
    'Recovered': 'sum',
    'Deaths': 'sum',
    'Active': 'sum'
}).reset_index()

# Computing daily new confirmed cases and recovery rate for India
india_data['Daily_New_Confirmed'] = india_data['Confirmed'].diff().fillna(0)
india_data['Daily_Recovered'] = india_data['Recovered'].diff().fillna(0)
india_data['Recovery_Rate'] = india_data['Recovered'] / india_data['Confirmed'].replace(0, np.nan)

# Visualizing Global Trends
# Plotting global daily new confirmed cases and recovery rate
fig_global = go.Figure()
fig_global.add_trace(go.Scatter(
    x=global_data['Date'],
    y=global_data['Daily_New_Confirmed'],
    mode='lines',
    name='Daily New Confirmed Cases (Global)'
))
fig_global.add_trace(go.Scatter(
    x=global_data['Date'],
    y=global_data['Daily_Recovered'],
    mode='lines',
    name='Daily Recovered (Global)'
))
fig_global.update_layout(
    title='Global COVID-19 Trends: Daily New Confirmed and Recovered Cases',
    xaxis_title='Date',
    yaxis_title='Count',
    legend=dict(x=0, y=1)
)
print("Global trends plot generated (visualization not displayed in text output).")

# Visualizing India Trends
fig_india = go.Figure()
fig_india.add_trace(go.Scatter(
    x=india_data['Date'],
    y=india_data['Daily_New_Confirmed'],
    mode='lines',
    name='Daily New Confirmed Cases (India)'
))
fig_india.add_trace(go.Scatter(
    x=india_data['Date'],
    y=india_data['Daily_Recovered'],
    mode='lines',
    name='Daily Recovered (India)'
))
fig_india.update_layout(
    title='India COVID-19 Trends: Daily New Confirmed and Recovered Cases',
    xaxis_title='Date',
    yaxis_title='Count',
    legend=dict(x=0, y=1)
)
print("India trends plot generated (visualization not displayed in text output).")

# Step 3: Time Series Forecasting with Prophet

# Preparing data for Prophet (Global)
prophet_global_data = global_data[['Date', 'Confirmed']].rename(columns={'Date': 'ds', 'Confirmed': 'y'})

# Fitting Prophet model for global data
model_global = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
model_global.fit(prophet_global_data)

# Creating future dates for the next 7 days
future_global = model_global.make_future_dataframe(periods=7)
forecast_global = model_global.predict(future_global)

# Preparing data for Prophet (India)
prophet_india_data = india_data[['Date', 'Confirmed']].rename(columns={'Date': 'ds', 'Confirmed': 'y'})

# Fitting Prophet model for India data
model_india = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
model_india.fit(prophet_india_data)

# Creating future dates for the next 7 days
future_india = model_india.make_future_dataframe(periods=7)
forecast_india = model_india.predict(future_india)

# Step 4: Visualize Predictions

# Plotting Global Forecast
fig_global_forecast = go.Figure()
fig_global_forecast.add_trace(go.Scatter(
    x=prophet_global_data['ds'],
    y=prophet_global_data['y'],
    mode='lines',
    name='Actual Confirmed (Global)'
))
fig_global_forecast.add_trace(go.Scatter(
    x=forecast_global['ds'],
    y=forecast_global['yhat'],
    mode='lines',
    name='Forecasted Confirmed (Global)'
))
fig_global_forecast.update_layout(
    title='Global COVID-19 Confirmed Cases: Actual vs Forecasted',
    xaxis_title='Date',
    yaxis_title='Confirmed Cases',
    legend=dict(x=0, y=1)
)
print("Global forecast plot generated (visualization not displayed in text output).")

# Plotting India Forecast
fig_india_forecast = go.Figure()
fig_india_forecast.add_trace(go.Scatter(
    x=prophet_india_data['ds'],
    y=prophet_india_data['y'],
    mode='lines',
    name='Actual Confirmed (India)'
))
fig_india_forecast.add_trace(go.Scatter(
    x=forecast_india['ds'],
    y=forecast_india['yhat'],
    mode='lines',
    name='Forecasted Confirmed (India)'
))
fig_india_forecast.update_layout(
    title='India COVID-19 Confirmed Cases: Actual vs Forecasted',
    xaxis_title='Date',
    yaxis_title='Confirmed Cases',
    legend=dict(x=0, y=1)
)
print("India forecast plot generated (visualization not displayed in text output).")

# Step 5: Print Forecast Results
# Global Forecast for the next 7 days
global_future_forecast = forecast_global.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print("\nGlobal Forecasted Confirmed Cases for the Next 7 Days:")
print(global_future_forecast)

# India Forecast for the next 7 days
india_future_forecast = forecast_india.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print("\nIndia Forecasted Confirmed Cases for the Next 7 Days:")
print(india_future_forecast)