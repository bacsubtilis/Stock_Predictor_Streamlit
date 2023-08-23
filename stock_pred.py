#########################################################
#                     Stock Predictor                   #
#                       by subtilis                     #
#                                                       #
#                   exercising for DAMA!                #
#########################################################

######## IMPORTS ###########
import streamlit as st, pandas as pd, numpy as np, yfinance as yf
import plotly.express as px

from prophet import Prophet
from prophet.plot import plot_plotly

from datetime import date

# Error handling
import urllib.request
import urllib.error

######## UI setup ########

START = date(1981, 1, 9) # The start date needed for the initialisation of the stock data
TODAY = date.today() # Needed for the max value of the end date below (otherwise we will not be able to select a date past 10 years of the start date)

st.title("Stock predictor")


ticker = st.sidebar.text_input('Ticker', value = 'AMGN')
start_date = st.sidebar.date_input('Start date', value= START, max_value= TODAY)
end_date = st.sidebar.date_input('End date')
n_years = st.sidebar.slider("Prediction period (years)", 1, 5)
period = n_years * 365

# Check if the ticker provided is correct. Otherwise throw custom error and exit
try:
    urllib.request.urlretrieve('https://query2.finance.yahoo.com/v6/finance/quoteSummary/' + ticker + '?modules=financialData&modules=quoteType&modules=defaultKeyStatistics&modules=assetProfile&modules=summaryDetail&ssl=true')
except urllib.error.HTTPError:
    st.error("Ticker not found! Please check your spelling or Yahoo Finance site for correct ticker name.")
    st.stop()

# Get the proper data for the ticker
ticker_complete = yf.Ticker(ticker)
company_name = ticker_complete.info['longName']

# Get the data based on user supplied ticker

@st.cache_data # Cache previous data to make things faster
def load_data(ticker):
    data = yf.download(ticker, start = start_date, end = end_date)
    data.reset_index(inplace=True) # Reset the index so that Date is not the index anymore and can be used
    return data

# Draw the first basic figure from the data with the based on the closing value over time
data_init = load_data(ticker=ticker)
fig = px.line(data_init, x = data_init['Date'], y = data_init['Adj Close'], title = company_name)
st.plotly_chart(fig)

##### Prediction based on Prophet #####
#######################################

# Create the training dataset
df_train = data_init[['Date', 'Adj Close']]
df_train = df_train.rename(columns = {"Date" : "ds", "Adj Close" : "y"}) # No need to convert the date (as in the initial notebook), no timezones imported with the download function

# Train Prophet
m = Prophet()
m.fit(df_train)

# Make the predicitons
future = m.make_future_dataframe(periods=period) # Here for the periods we use the used defined period from the slider in the UI (from 1 to 5 years)
forecast = m.predict(future)

# Show the figure with the predictions
pred_fig_prophet = plot_plotly(m, forecast)

st.markdown("## Predictions")
st.markdown("### Raw forecast data")
st.write(forecast)
st.markdown("### Forecast chart")
st.plotly_chart(pred_fig_prophet)

# Show the individual components of the predictions
st.markdown("### Forecast components")
comp_fig_prophet = m.plot_components(forecast)
st.write(comp_fig_prophet)

############################################# TODO ########################################################
# 1. Create drop down list for the user to choose from the available column values for the analysis
# 2. Create a multipage app for each model used (if in the future the Decision tree model is implemented)
# 3. Show only the forecasted data for the time period selected (e.g. 1 year)
# 4. 