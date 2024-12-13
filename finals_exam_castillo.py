import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Streamlit App Title
st.title("ARIMA Model Time Series Forecasting")

# Define file paths
co2_file_path = "data.csv"
temp_file_path = "dataset_temperature.csv"

# Load CO2 data
try:
    co2_data = pd.read_csv(co2_file_path, parse_dates=['Month'], index_col='Month')
    co2_data = co2_data.asfreq('M')

    # Visualize the CO2 time series
    st.subheader("CO2 Levels Time Series")
    st.line_chart(co2_data['CO2 (ppm)'])

    # Fit baseline ARIMA model
    co2_model = ARIMA(co2_data, order=(1, 1, 1))
    co2_fit = co2_model.fit()

    # Forecast the next 10 steps
    co2_forecast = co2_fit.forecast(steps=10)
    st.subheader("CO2 Forecast")
    st.write(co2_forecast)

    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(co2_data, label='Historical CO2 Levels', color='blue')
    plt.plot(pd.date_range(start=co2_data.index[-1], periods=11, freq='M')[1:], co2_forecast, label='Forecasted CO2 Levels', color='orange')
    plt.title('CO2 Forecast')
    plt.xlabel('Date')
    plt.ylabel('CO2 (ppm)')
    plt.legend()
    st.pyplot(plt)
except FileNotFoundError:
    st.error("CO2 data file not found. Please check the file path.")

# Load temperature data
try:
    temp_data = pd.read_csv(temp_file_path)
    
    # Debug: Show the first few rows and datatypes
    st.write("Temperature Data Preview:")
    st.write(temp_data.head())
    st.write("Data Types:")
    st.write(temp_data.dtypes)

    # Convert 'Month' column to datetime
    temp_data['Month'] = pd.to_datetime(temp_data['Month'], errors='coerce')
    temp_data.set_index('Month', inplace=True)
    temp_data.columns = temp_data.columns.str.strip()  # Clean column names

    # Check if the required column exists
    if 'Mean monthly temperature' not in temp_data.columns:
        st.error("Column 'Mean monthly temperature' not found in the dataset.")
    else:
        # Convert to numeric and handle missing values
        temp_data['Mean monthly temperature'] = pd.to_numeric(temp_data['Mean monthly temperature'], errors='coerce')
        temp_data.dropna(inplace=True)  # Drop missing values

        # Check if temp_data is empty
        if temp_data.empty:
            st.error("Temperature data is empty. Please check the file.")
        else:
            # Debug: Print the index and the last date
            st.write("Temperature Data Index:")
            st.write(temp_data.index)
            st.write("Last Date in Temperature Data Index:")
            st.write(temp_data.index[-1])

            # Visualize temperature data
            st.subheader("Temperature Levels Time Series")
            st.line_chart(temp_data['Mean monthly temperature'])

            # Fit baseline ARIMA model
            baseline_model = ARIMA(temp_data['Mean monthly temperature'], order=(1, 1, 1))
            baseline_results = baseline_model.fit()

            # Forecast the next 12 months
            try:
                forecast = baseline_results.get_forecast(steps=12)
                last_date = temp_data.index[-1]
                st.write("Last Date for Forecasting:")
                st.write(last_date)

                forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='M')
                forecast_values = forecast.predicted_mean
                forecast_conf_int = forecast.conf_int()

                # Plot the forecasts
                plt.figure(figsize=(12, 6))
                plt.plot(temp_data.index, temp_data['Mean monthly temperature'], label='Historical Data', color='blue')
                plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
                plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
                plt.title('Temperature Forecast')
                plt.xlabel('Date')
                plt.ylabel('Temperature (°C)')
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(plt)
            except Exception as e:
                st.error(f"An error occurred while generating the forecast: {e}")
except FileNotFoundError:
    st.error("Temperature data file not found. Please check the file path.")

# Perform grid search for best ARIMA parameters
if st.button("Find Best ARIMA Parameters for Temperature Data"):
    if 'temp_data' in locals() and not temp_data.empty:  # Ensure temp_data is loaded and not empty
        p = d = q = range(0, 3)  # Adjust ranges as necessary
        pdq = list(itertools.product(p, d, q))

        best_aic = float('inf')
        best_pdq = None

        for param in pdq:
            try:
                temp_model = ARIMA(temp_data['Mean monthly temperature'], order=param)
                results = temp_model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_pdq = param
            except Exception as e:
                continue

        st.write(f'Best ARIMA parameters: {best_pdq} with AIC: {best_aic}')
    else:
        st.warning("Temperature data not loaded or is empty. Please check the file path.")
