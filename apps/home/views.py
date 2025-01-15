# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from django.contrib import messages
from django.shortcuts import render, redirect
from .forms import *
from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import loader
from django.urls import reverse
from .forms import *
from .models import *
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup
import seaborn as sns
import warnings
from statsmodels.tsa.arima.model import ARIMA
import sys
import yfinance as yf
import numpy as np
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, Dense
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import io
from contextlib import contextmanager
from datetime import datetime
import pmdarima as pm
from django.shortcuts import render, redirect
from .models import *
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError
from django.conf import settings
print(settings.DATABASES)
from django.db import connection
print(connection.settings_dict)
import logging
logger = logging.getLogger(__name__)
import yfinance as yf
from celery import shared_task
from .models import Portfolio
from decimal import Decimal
from .tasks import *




  # You can dynamically set this symbol

@login_required(login_url="/login/")
def index(request):
    update_portfolio_performance()
    hourly_portfolio_snapshot()
    
    symbol = get_ticker(request)
    news_data = fetch_news(symbol)
    data=fetch_data_from_yf(symbol)
    data=calculate_indicators(data)
    explanations=explain_indicators(data)
    macro_data,macro_suggestion=analyze_macroeconomic_data()

    context = {
        'segment': 'index',
        'news_data': news_data,
        'data': data,
        'macro_data': macro_data,
        'macro_suggestion': macro_suggestion,
        'explanations':explanations
    }

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))


def get_ticker(request):
    ticker = request.GET.get('ticker')  # Get the ticker from the request
    # Example: Perform your logic based on the ticker
    return ticker.upper() if ticker else 'AAPL'




        

def fetch_data_from_yf(symbol,period='1y',interval='1d'):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df.empty:
            print(f"No data returned for {symbol}")
            return None

        df['symbol'] = symbol
        df['Return'] = df['Adj Close'].pct_change()
        df.reset_index(inplace=True)
        df.rename(columns={
            'Datetime': 'timestamp',
            'Date': 'timestamp',
            'timestamp': 'timestamp',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj Close',
            'Volume': 'volume'
        }, inplace=True)

        def fetch_pe_ratio(symbol):
            stock = yf.Ticker(symbol)
            try:
                info = stock.info
                pe_ratio = info.get('forwardEps') / info.get('currentPrice') if info.get('currentPrice') else None
                return pe_ratio
            except Exception as e:
                print(f"Error fetching P/E ratio for {symbol}: {e}")
                return None

        df['pe-ratio']=fetch_pe_ratio(symbol)

        return df
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance for {symbol}: {e}")
        return None
  

def fetch_data_from_yf_pred(symbol,period='1y',interval='1d'):
    try:
        # Download data from Yahoo Finance
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df.empty:
            print(f"No data returned for {symbol}")
            return None

        df['symbol'] = symbol
        df['Return'] = df['Adj Close'].pct_change()

        # Ensure the index is a DatetimeIndex and reset it into a column
        df.reset_index(inplace=True)  # Reset index to make 'Date' a column
        
        # Ensure the 'timestamp' is correctly renamed and in datetime format
        df.rename(columns={'Date': 'timestamp'}, inplace=True)

        # Convert 'timestamp' column to datetime, in case it's not in the right format
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Drop rows where 'timestamp' could not be parsed
        df.dropna(subset=['timestamp'], inplace=True)

        return df
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance for {symbol}: {e}")
        return None    

# News Sensitivity Analysis
def fetch_news(symbol):
    stock = yf.Ticker(symbol)
    news_list = stock.news
    if not news_list:
        return []

    news_data = []
    for news_item in news_list:
        thumbnail_url = None
        if 'thumbnail' in news_item and 'resolutions' in news_item['thumbnail']:
            # Extract the URL of the first resolution
            resolutions = news_item['thumbnail']['resolutions']
            if resolutions:
                thumbnail_url = resolutions[0].get('url')

        news_data.append({
            'title': news_item.get('title'),
            'url': news_item.get('link'),
            'thumbnail': thumbnail_url
        })

    return news_data

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return {
        'neg': analysis.sentiment.polarity < 0,
        'neu': analysis.sentiment.polarity == 0,
        'pos': analysis.sentiment.polarity > 0,
        'compound': analysis.sentiment.polarity
    }



# Fetch news and sentiment data (CoinMarketCap or any suitable API)
def fetch_news_sentiment():
    print('Process: fetch_news_sentiment')
    url = "https://newsapi.org/v2/everything?q=stock market&apiKey=3506c8c7d53d411c97371fe60cd2c050"
    response = requests.get(url)
    news_data = response.json().get('articles', [])
    print(news_data)

    sentiments = []
    for news in news_data:
      sentiment = analyze_sentiment(news['description'] or '')
      sentiments.append(sentiment['compound'])
      print(sentiment)
    return np.mean(sentiments) #if sentiments else 0



def get_news_sentiment(symbol):
    print('Process: get_news_sentiment')
    news_data = fetch_news(symbol)
    if not news_data:
        return "No news available for this symbol."

    sentiment_results = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for new in news_data:
        print(f"Title: {new['title']} and URL: {new['url']}")
        sentiment = analyze_sentiment(new["title"])  # This now returns a dict

        # Determine sentiment label
        if sentiment['pos']:
            sentiment_label = 'Positive'
        elif sentiment['neg']:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'

        print('Sentiment data:', sentiment_label)  # This should print the sentiment label

        # Update sentiment count
        if sentiment_label in sentiment_results:
            sentiment_results[sentiment_label] += 1
        else:
            print(f"Unknown sentiment: {sentiment_label}")

    # Determine the most dominant sentiment
    max_sentiment = max(sentiment_results, key=sentiment_results.get)
    print(f'Max sentiment: {max_sentiment}')
    return max_sentiment


  
def calculate_indicators(df):
    if df.empty:
        return df

    print(df.columns)
    # Moving Average
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()

    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BBM_20'] = df['Close'].rolling(window=20).mean()
    df['BBU_20'] = df['BBM_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BBL_20'] = df['BBM_20'] - 2 * df['Close'].rolling(window=20).std()

    # Average True Range (ATR)
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = np.abs(df['High'] - df['Close'].shift())
    df['Low-Close'] = np.abs(df['Low'] - df['Close'].shift())
    df['True_Range'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['ATR'] = df['True_Range'].rolling(window=14).mean()

    return df    

def analyze_macroeconomic_data():
    # MacroEconomics Indicators
    fred = Fred(api_key='38a85b8262a0044479cc6200b3c2c99f')
    macro_data = {
        'gdp_data': {},
        'cpi_data': {},
        'ppi_data': {},
        'unemployment_data': {},
        'fed_funds_rate': {},
        'consumer_confidence_data': {},
        'pmi_data': {}
    }
    positive_indicators = 0
    negative_indicators = 0
    macro_data['gdp_data']['suggestion']='Neutral'
    macro_data['cpi_data']['suggestion']='Neutral'
    macro_data['ppi_data']['suggestion']='Neutral'
    macro_data['unemployment_data']['suggestion']='Neutral'
    macro_data['fed_funds_rate']['suggestion']='Neutral'
    macro_data['consumer_confidence_data']['suggestion']='Neutral'
    macro_data['pmi_data']['suggestion']='Neutral'
    macro_data['gdp_data']['description']='Total economic output (Higher is better)'
    macro_data['cpi_data']['description']='Inflation measure (Stable or low is better)'
    macro_data['ppi_data']['description']='Producer price index (Stable is better)'
    macro_data['unemployment_data']['description']='Employment measure (Lower is better)'
    macro_data['fed_funds_rate']['description']='Interest rate (Lower is better)'
    macro_data['consumer_confidence_data']['description']='Consumer outlook (Higher is better)'
    macro_data['pmi_data']['description']='Economic growth indicator (Above 50 is better)'


    macro_data['gdp_data']['url']='https://data.worldbank.org/indicator/NY.GDP.MKTP.CD'
    macro_data['cpi_data']['url']='https://www.investing.com/economic-calendar/cpi-733?utm_source=google&utm_medium=cpc&utm_campaign=21449641878&utm_content=705317573943&utm_term=dsa-1456167871416_&GL_Ad_ID=705317573943&GL_Campaign_ID=21449641878&ISP=1&npl=1&ppu=9801673&gad_source=1&gclid=Cj0KCQjw3bm3BhDJARIsAKnHoVUZ4h_aKU4JNSY9sZ0NdNxA0Re-lBBh3XfSHKucR3kxab7VFeyKjJMaAhvzEALw_wcB'
    macro_data['ppi_data']['url']='https://www.investing.com/economic-calendar/ppi-734?utm_source=google&utm_medium=cpc&utm_campaign=18502377045&utm_content=626070285991&utm_term=dsa-1546555491774_&GL_Ad_ID=626070285991&GL_Campaign_ID=18502377045&ISP=1&gad_source=1&gclid=Cj0KCQjw3bm3BhDJARIsAKnHoVWQLCa9GlyxOVTdedtRAnumPtaeNJhF3UEG36MZXCvm6PYda6CSWVAaAkoGEALw_wcB'
    macro_data['unemployment_data']['url']='https://fred.stlouisfed.org/series/UNRATE'
    macro_data['fed_funds_rate']['url']='https://fred.stlouisfed.org/series/FEDFUNDS'
    macro_data['consumer_confidence_data']['url']='https://tradingeconomics.com/country-list/consumer-confidence'
    macro_data['pmi_data']['url']='https://tradingeconomics.com/united-states/business-confidence'
    # Fetching GDP data (GDP is the identifier for the Gross Domestic Product in FRED)
    try:
        gdp_data = fred.get_series('GDP')
        macro_data['gdp_data']['value'] = gdp_data.iloc[-1]
        macro_data['gdp_data']['previous_value'] = gdp_data.iloc[-2]
    # GDP is generally considered a positive indicator
        if macro_data.get('gdp_data'):
          macro_data['gdp_data']['suggestion']='Positive'
          positive_indicators += 1
    except Exception as e:
        print(f"Error fetching GDP data: {e}")
        gdp_data = None

    # Inflation Rate (CPI/PPI)
    try:
        cpi_data = fred.get_series('CPIAUCNS')
        macro_data['cpi_data']['value'] = cpi_data.iloc[-1]
        macro_data['cpi_data']['previous_value'] = cpi_data.iloc[-2]
    # CPI and PPI: Inflationary indicators; High values can be negative
        if macro_data.get('cpi_data') and macro_data['cpi_data'] > 2:
            macro_data['cpi_data']['suggestion']='Negative'
            negative_indicators += 1
        if macro_data.get('ppi_data') and macro_data['ppi_data'] > 2:
            negative_indicators += 1
            macro_data['ppi_data']['suggestion']='Negative'
    except Exception as e:
        cpi_data = None

    try:
        ppi_data = fred.get_series('PPIACO')
        macro_data['ppi_data']['value'] = ppi_data.iloc[-1]
        macro_data['ppi_data']['previous_value'] = ppi_data.iloc[-2]
    except Exception as e:
        ppi_data = None

    # Unemployment Rate
    try:
        unemployment_data = fred.get_series('UNRATE')
        macro_data['unemployment_data']['value'] = unemployment_data.iloc[-1]
        macro_data['unemployment_data']['previous_value'] = unemployment_data.iloc[-2]
        # Unemployment Rate: Lower is generally better
        if macro_data.get('unemployment_data') and macro_data['unemployment_data'] < 5:
            positive_indicators += 1
            macro_data['unemployment_data']['suggestion']='Positive'
        else:
            negative_indicators += 1
            macro_data['unemployment_data']['suggestion']='Negative'
    except Exception as e:
        unemployment_data = None

    # Interest Rates
    try:
        fed_funds_rate = fred.get_series('FEDFUNDS')
        macro_data['fed_funds_rate']['value'] = fed_funds_rate.iloc[-1]
        macro_data['fed_funds_rate']['previous_value'] = fed_funds_rate.iloc[-2]
        # Federal Funds Rate: Higher rates are often seen as negative
        if macro_data.get('fed_funds_rate') and macro_data['fed_funds_rate'] < 2:
            positive_indicators += 1
            macro_data['fed_funds_rate']['suggestion']='Positive'
        else:
            negative_indicators += 1
            macro_data['fed_funds_rate']['suggestion']='Negative'
    except Exception as e:
        fed_funds_rate = None

    # Consumer Confidence Index
    try:
        consumer_confidence_data = fred.get_series('CONCCONF')
        macro_data['consumer_confidence_data']['value'] = consumer_confidence_data.iloc[-1]
        macro_data['consumer_confidence_data']['previous_value'] = consumer_confidence_data.iloc[-2]
        # Consumer Confidence Index: Higher is better
        if macro_data.get('consumer_confidence_data') and macro_data['consumer_confidence_data'] > 100:
            positive_indicators += 1
            macro_data['consumer_confidence_data']['suggestion']='Positive'
        else:
            macro_data['consumer_confidence_data']['suggestion']='Neutral'    
    except Exception as e:
        consumer_confidence_data = None

    # PMI (Purchasing Managers' Index)
    try:
        pmi_data = fred.get_series('ISM/MAN_PMI')
        macro_data['pmi_data']['value'] = pmi_data.iloc[-1]
        macro_data['pmi_data']['previous_value'] = pmi_data.iloc[-2]
            # PMI: Higher indicates economic growth
        if macro_data.get('pmi_data') and macro_data['pmi_data'] > 50:
            positive_indicators += 1
            macro_data['pmi_data']['suggestion']='Positive'
        else:
            negative_indicators += 1
            macro_data['pmi_data']['suggestion']='Negative'
    except Exception as e:
        pmi_data = None

    if not macro_data:
        return "Neutral"

    if positive_indicators > negative_indicators:
        suggestion="Macroeconomic Suggestion: Positive"
    elif negative_indicators > positive_indicators:
        suggestion="Macroeconomic Suggestion: Negative"
    else:
        suggestion="Macroeconomic Suggestion: Neutral"
    print(macro_data)
    return macro_data,suggestion


def get_chartBig1_data(request):
    try:
        symbol = request.GET.get('ticker', 'AAPL')  # Default to 'GOOGL'
        period = request.GET.get('period', '1y')  # Default to '1y'
        interval = request.GET.get('interval', '1mo')  # Default to '1mo'
        
        # Fetch data
        data = fetch_data_from_yf(symbol, period, interval)
        print("Fetched Data:", data)

        # Check if the data is valid
        if isinstance(data, pd.DataFrame) and 'timestamp' in data and 'Close' in data:
            labels = data['timestamp'].dt.strftime('%d/%m/%Y').tolist()
            close_prices = data['Close'].tolist()
        else:
            return JsonResponse({'error': 'Invalid data format or missing columns'}, status=400)

        # Prepare the chart data
        chart_data = {
            'symbol': symbol,
            'labels': labels,
            'data': close_prices,
            'last_price':round(close_prices[-1], 2)
        }

        return JsonResponse(chart_data)

    except Exception as e:
        print("Error:", e)
        return JsonResponse({'error': str(e)}, status=500)
    

def get_chartpurple_data(request):
    try:
        symbol = '^GSPC' #request.GET.get('ticker', '^GSPC')  # Default to 'GOOGL'
        period = request.GET.get('period', '1y')  # Default to '1y'
        interval = request.GET.get('interval', '1d')  # Default to '1mo'
        
        # Fetch data
        data = fetch_data_from_yf(symbol, period, interval)
        print("Fetched Data:", data)

        # Check if the data is valid
        if isinstance(data, pd.DataFrame) and 'timestamp' in data and 'Close' in data:
            labels = data['timestamp'].dt.strftime('%d/%m/%Y').tolist()
            close_prices = data['Close'].tolist()
        else:
            return JsonResponse({'error': 'Invalid data format or missing columns'}, status=400)

        # Prepare the chart data
        chart_data = {
            'labels': labels,
            'data': close_prices,
            'last_price':round(close_prices[-1], 2)
        }

        return JsonResponse(chart_data)

    except Exception as e:
        print("Error:", e)
        return JsonResponse({'error': str(e)}, status=500)    

def create_charts(request):
    period = request.GET.get('period', '1y')
    interval = request.GET.get('interval', '1d')
    symbol = request.GET.get('ticker', 'GOOGL')
    data = fetch_data_from_yf(symbol, period, interval)
    if data is None or data.empty:
        print(f"No data available for {symbol} with period {period} and interval {interval}")
        return


    # Calculate indicators
    data = calculate_indicators(data)

    # Create traces for plotly
    traces = []

    # Closing price trace
    traces.append(go.Scatter(x=data['timestamp'], y=data['Close'], mode='lines', name='Close', line=dict(color='#00B2E2')))

    # Bollinger Bands traces
    traces.append(go.Scatter(x=data['timestamp'], y=data['BBU_20'], mode='lines', name='Bollinger Upper Band', line=dict(color='red', dash='dash')))
    traces.append(go.Scatter(x=data['timestamp'], y=data['BBL_20'], mode='lines', name='Bollinger Lower Band', line=dict(color='red', dash='dash')))

    # Moving Averages traces
    traces.append(go.Scatter(x=data['timestamp'], y=data['SMA_20'], mode='lines', name='MA 20', line=dict(color='green')))
    traces.append(go.Scatter(x=data['timestamp'], y=data['SMA_50'], mode='lines', name='MA 50', line=dict(color='orange')))
    traces.append(go.Scatter(x=data['timestamp'], y=data['SMA_200'], mode='lines', name='MA 200', line=dict(color='purple')))

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout for interactive features
    fig.update_layout(
        title=f'{symbol} - {period} {interval}',
        xaxis_title=f'Date: {data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")}',
        yaxis_title='Value',
        template='plotly_dark',  # Dark theme
        hovermode='x unified'
    )

    # Show the plot
    fig.show()



#Prediction models chart
def fetch_data_from_yf_predic(symbol, period, interval):
    data = yf.download(symbol, interval=interval, period=period, progress=False)
    data = data[['Adj Close']]
    return data

# ARIMA Model
def arima_model(stock_data):
    try:
        auto_arima_model = pm.auto_arima(
            stock_data['Adj Close'],
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            d=1,
            seasonal=False,
            stepwise=True,
            trace=True,
            suppress_warnings=True,
            error_action='ignore',
            approx=True
        )
        arima_result = auto_arima_model.fit(stock_data['Adj Close'])

        # In-sample predictions
        stock_data['ARIMA_Prediction'] = arima_result.predict_in_sample(dynamic=False)

        # Calculate future predictions
        last_date = stock_data.index[-1]
        today = pd.Timestamp(datetime.today().date())
        future_days = (today - last_date).days

        if future_days > 0:
            future_forecast = arima_result.predict(n_periods=future_days)
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
            future_df = pd.DataFrame(future_forecast, index=future_dates, columns=['ARIMA_Prediction'])
            stock_data = pd.concat([stock_data, future_df])
            print(f"Future predictions made up to {today}")
        else:
            print("No future prediction needed.")

    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")

    return stock_data, arima_result



# LSTM Model
def lstm_model(stock_data):
    # Scale the data
    data = stock_data['Adj Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare LSTM input
    X = []
    y = []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Define and compile the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, batch_size=1, epochs=10)

    # Generate predictions
    lstm_predictions_scaled = model.predict(X)
    lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

    # Add predictions to the DataFrame
    stock_data['LSTM_Prediction'] = np.nan
    stock_data.iloc[60:, stock_data.columns.get_loc('LSTM_Prediction')] = lstm_predictions.reshape(-1)
    return stock_data,  lstm_predictions


def plot_prediction(request):
    period = request.GET.get('period')
    interval = request.GET.get('interval')
    symbol = request.GET.get('ticker', 'GOOGL')
    stock_data = fetch_data_from_yf(symbol='AAPL', period='1y', interval='1d')
    stock_data,  lstm_predictions = lstm_model(stock_data)
    stock_data, arima_result = arima_model(stock_data)
    train_size = int(len(stock_data) * 0.8)
    train_data = stock_data[:train_size]
    test_data = stock_data[train_size:].copy()

    # ARIMA predictions
    arima_pred = arima_result.predict_in_sample(start=len(train_data), end=len(stock_data)-1)

    # Future ARIMA predictions
    future_steps = len(test_data)
    arima_future_pred = arima_result.predict(n_periods=future_steps)

    # Future LSTM predictions
    future_lstm_predictions = np.full(future_steps, np.nan)
    if len(lstm_predictions) >= future_steps:
        future_lstm_predictions[:future_steps] = lstm_predictions.flatten()[:future_steps]

    # Create future DataFrame
    last_date = pd.to_datetime(test_data.index[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
    future_df = pd.DataFrame(index=future_dates, data={
        'ARIMA_Prediction': arima_future_pred,
        'LSTM_Prediction': future_lstm_predictions
    })

    # Plotting
    traces = []

    traces.append(go.Scatter(x=train_data.index, y=train_data['Adj Close'], mode='lines', name='Train Data', line=dict(color='#00B2E2')))
    traces.append(go.Scatter(x=test_data.index, y=test_data['Adj Close'], mode='lines', name='Test Data', line=dict(color='#FF6600')))
    traces.append(go.Scatter(x=test_data.index, y=arima_pred, mode='lines', name='ARIMA Predictions', line=dict(color='purple', dash='dash')))
    traces.append(go.Scatter(x=test_data.index, y=test_data['LSTM_Prediction'], mode='lines', name='LSTM Predictions', line=dict(color='green', dash='dash')))

    if not future_df['ARIMA_Prediction'].isna().all():
        traces.append(go.Scatter(x=future_df.index, y=future_df['ARIMA_Prediction'], mode='lines', name='Future ARIMA Predictions', line=dict(color='purple', dash='dot')))
    if not future_df['LSTM_Prediction'].isna().all():
        traces.append(go.Scatter(x=future_df.index, y=future_df['LSTM_Prediction'], mode='lines', name='Future LSTM Predictions', line=dict(color='green', dash='dot')))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Stock Price Prediction: ARIMA & LSTM',
        xaxis_title='Date',
        yaxis_title='Close Price (USD)',
        template='plotly_dark',
        hovermode='x unified'
    )

    fig.show()

'''
def get_CountryChart_data(request):
    print('***************************************')
    print('***************************************')        
    try:
        symbol = request.GET.get('ticker', 'AAPL')  # Default to 'AAPL'
        period = request.GET.get('period', '1y')  # Default to '1y'
        interval = request.GET.get('interval', '1mo')  # Default to '1mo'
        
        # Fetch data
        stock_data = fetch_data_from_yf(symbol, period, interval)
        print("Fetched Data:", stock_data)

        # Check if the data is valid
        if isinstance(stock_data, pd.DataFrame) and 'timestamp' in stock_data and 'Close' in stock_data:
            labels = stock_data['timestamp'].dt.strftime('%d/%m/%Y').tolist()
            close_prices = stock_data['Close'].tolist()
        else:
            return JsonResponse({'error': 'Invalid data format or missing columns'}, status=400)


        print('######## Debugging #########')
        print(stock_data.head())            # Display the first few rows of the DataFrame
        print(stock_data.index)             # Print the index to check its type
        print(stock_data.info())            # Get a summary of the DataFrame including data types
        print('#### End of Debugging #####')

        # LSTM and ARIMA model predictions
        stock_data, lstm_predictions = lstm_model(stock_data)
        stock_data, arima_result = arima_model(stock_data)

        train_size = int(len(stock_data) * 0.8)
        train_data = stock_data[:train_size]

        # ARIMA predictions
        arima_pred = arima_result.predict_in_sample(start=len(train_data), end=len(stock_data)-1)
        future_steps = len(stock_data) - train_size
        arima_future_pred = arima_result.predict(n_periods=future_steps)

        # Prepare future DataFrame with ARIMA and LSTM predictions
        future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
        future_df = pd.DataFrame({
            'Date': future_dates,
            'ARIMA_Prediction': arima_future_pred,
            'LSTM_Prediction': lstm_predictions.flatten()[:future_steps] if lstm_predictions.size >= future_steps else np.full(future_steps, np.nan)
        })

        # Ensure Date column is in correct format for chart labels
        if isinstance(future_df, pd.DataFrame) and 'Date' in future_df and 'ARIMA_Prediction' in future_df and 'LSTM_Prediction' in future_df:
            labels = future_df['Date'].strftime('%d/%m/%Y').tolist()  # Convert to list of string dates
            arima = future_df['ARIMA_Prediction'].tolist()  # Ensure it's a list
            lstm = future_df['LSTM_Prediction'].tolist()  # Ensure it's a list
        else:
            return JsonResponse({'error': 'Invalid data format or missing columns'}, status=400)

        # Prepare the chart data
        chart_data = {
            "labels": labels,
            "arima": arima,
            "lstm": lstm
        }

        return JsonResponse(chart_data)

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Log the error
        return JsonResponse({'error': str(e)}, status=500)
    
'''
def get_predictions(request):
    # Fetch stock data
    symbol = 'AAPL'  # Example stock symbol, you can make it dynamic
    period = '1y'
    interval = '1d'
    stock_data = fetch_data_from_yf_predic(symbol, period, interval)

    # Apply ARIMA model
    stock_data, arima_predictions = arima_model(stock_data)

    # Apply LSTM model
    stock_data, lstm_predictions = lstm_model(stock_data)

    # Prepare data for response
    data = {
        'arima_predictions': stock_data['ARIMA_Prediction'].dropna().tolist(),
        'lstm_predictions': stock_data['LSTM_Prediction'].dropna().tolist(),
    }

    return JsonResponse(data)


def get_CountryChart_data(request):
    try:
        symbol = request.GET.get('ticker', 'AAPL')  # Default to 'AAPL'
        period = request.GET.get('period', '1y')  # Default to '1y'
        interval = request.GET.get('interval', '1mo')  # Default to '1mo'
            
        # Fetch data
        stock_data = fetch_data_from_yf_predic(symbol, period, interval)
        stock_data.index = pd.to_datetime(stock_data.index, errors='coerce')  # Ensure datetime index

        # Add empty LSTM_Prediction column
        stock_data.loc[:, 'LSTM_Prediction'] = np.nan

        # LSTM and ARIMA model predictions
        stock_data, lstm_predictions = lstm_model(stock_data)
        stock_data, arima_result = arima_model(stock_data)

        # Log the length of stock data and predictions
        print(f"Length of stock_data: {len(stock_data)}")
        print(f"Length of LSTM predictions: {len(lstm_predictions)}")

        # Add ARIMA predictions
        stock_data.loc[:, 'ARIMA_Prediction'] = arima_result.predict_in_sample(dynamic=False)
        stock_data.set_index('timestamp', inplace=True)  # Replace 'date_column' with your actual date column name


        train_size = int(len(stock_data) * 0.8)
        train_data = stock_data[:train_size]

        # ARIMA future predictions
        future_steps = len(stock_data) - train_size
        arima_future_pred = arima_result.predict(n_periods=future_steps)

        # Log the length of ARIMA future predictions
        print(f"Length of ARIMA future predictions: {len(arima_future_pred)}")
        
        # Prepare future DataFrame with ARIMA and LSTM predictions
        future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
        future_df = pd.DataFrame({
            'Date': future_dates,
            'ARIMA_Prediction': arima_future_pred,
            'LSTM_Prediction': lstm_predictions.flatten()[:future_steps] if lstm_predictions.size >= future_steps else np.full(future_steps, np.nan)
        })

        # Check if the future dataframe was created properly
        print(f"Future DataFrame created with {len(future_df)} rows.")

        # Convert 'Date' to proper datetime format for chart labels
        future_df['Date'] = pd.to_datetime(future_df['Date'], errors='coerce')
        labels = future_df['Date'].dt.strftime('%d/%m/%Y').tolist()  # Use .dt for datetime functions
        arima = future_df['ARIMA_Prediction'].tolist()  # Ensure it's a list
        lstm = future_df['LSTM_Prediction'].tolist()  # Ensure it's a list

        # Check if lists are populated
        print(f"Labels: {labels}")
        print(f"ARIMA predictions: {arima}")
        print(f"LSTM predictions: {lstm}")

        # Prepare the chart data
        chart_data = {
            "labels": labels,
            "arima": arima,
            "lstm": lstm
        }

        return JsonResponse(chart_data)

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Log the error
        data = fetch_data_from_yf(symbol, period, interval)
        print("Fetched Data:", data)

        # Check if the data is valid
        if isinstance(data, pd.DataFrame) and 'timestamp' in data and 'Close' in data:
            labels = data['timestamp'].dt.strftime('%d/%m/%Y').tolist()
            close_prices = data['Close'].tolist()
        else:
            return JsonResponse({'error': 'Invalid data format or missing columns'}, status=400)

        # Prepare the chart data
        chart_data = {
            'symbol': symbol,
            'labels': labels,
            'data': close_prices,
            'last_price':round(close_prices[-1], 2)
        }
        return JsonResponse(chart_data)


def explain_indicators(df, source=""):

    explanations = {}

    def add_explanation(indicator, value, explanation, feeling):
        explanations[indicator] = {
            'value': value,
            'description': explanation,
            'feeling': feeling
        }

    """
    Explain the indicators and provide a rationale for whether the signal is Positive, Negative, or Neutral.
    """
    latest_data = df.tail().iloc[-1]
    print(f"\n{source} Indicator Explanation:")

    # Moving Averages
    print(f"SMA_20: {latest_data['SMA_20']:.2f} (20-day Moving Average)")
    add_explanation('SMA_20', round(latest_data['SMA_20'],2),
        "The current price is above the 20-day moving average- Positive signal.",
        "Positive" if latest_data['Close'] > latest_data['SMA_20'] else "Negative")

    if latest_data['Close'] > latest_data['SMA_20']:
        print("The current price is above the 20-day moving average- Positive signal.")
    else:
        print("The current price is beLow the 20-day moving average- Negative signal.")

    # RSI
    if latest_data['RSI'] < 30:
        add_explanation('RSI', round(latest_data['RSI'],2), "RSI is beLow 30, stock might be oversold- Positive signal.", "Positive")
    elif latest_data['RSI'] > 70:
        add_explanation('RSI', round(latest_data['RSI'],2), "RSI is above 70, stock might be overbought- Negative signal.", "Negative")
    else:
        add_explanation('RSI', round(latest_data['RSI'],2), "RSI is between 30 and 70- Neutral stance.", "Neutral")

    print(f"RSI: {latest_data['RSI']:.2f} (Relative Strength Index)")
    
    # MACD
    print(f"MACD: {latest_data['MACD']:.2f}")
    add_explanation('MACD', round(latest_data['MACD'],2), 
        "MACD is above the signal line,- Positive momentum signal.",
        "Positive" if latest_data['MACD'] > latest_data['MACD_Signal'] else "Negative")

    print(f"MACD Signal Line: {latest_data['MACD_Signal']:.2f}")


    # Bollinger Bands
    print(f"Bollinger Upper Band: {latest_data['BBU_20']:.2f}")
    print(f"Bollinger Lower Band: {latest_data['BBU_20']:.2f}")
    if latest_data['Close'] < latest_data['BBL_20']:
        add_explanation('Bollinger_Lower', round(latest_data['Close'],2), "Price is beLow Lower Bollinger Band, -Positive buying opportunity.", "Positive")
    elif latest_data['Close'] > latest_data['BBU_20']:
        add_explanation('Bollinger_Upper', round(latest_data['Close'],2), "Price is above upper Bollinger Band- overbought condition- Negative.", "Negative")
    else:
        add_explanation('Bollinger_Upper', round(latest_data['Close'],2), "Price is within the Bollinger Bands, Neutral outlook.", "Neutral")
        add_explanation('Bollinger_Lower', round(latest_data['Close'],2), "Price is within the Bollinger Bands, Neutral outlook.", "Neutral")

    # ATR (Volatility)
    print(f"ATR: {latest_data['ATR']:.2f} (Average True Range)")
    add_explanation('ATR', round(latest_data['ATR'],2), 
        "High ATR suggests increased volatility, which can be a risk factor.",
        "Negative" if latest_data['ATR'] > df['ATR'].mean() else "Positive")

    if latest_data['ATR'] > df['ATR'].mean():
        print("High ATR suggests increased volatility, which can be a risk factor.")
    else:
        print("Low ATR suggests Lower volatility, which could imply stability.")

    # Average True Range Components
    print(f"High-Low: {latest_data['High-Low']:.2f}")
    print(f"High-Close: {latest_data['High-Close']:.2f}")
    print(f"Low-Close: {latest_data['Low-Close']:.2f}")
    print(f"True Range: {latest_data['True_Range']:.2f}")
    add_explanation('High-Low', latest_data['High-Low'], "Difference between the High and Low prices of the day.", "Neutral")
    add_explanation('High-Close', latest_data['High-Close'], "Difference between the High price and the previous Close price.", "Neutral")
    add_explanation('Low-Close', latest_data['Low-Close'], "Difference between the Low price and the previous Close price.", "Neutral")
    add_explanation('True_Range', latest_data['True_Range'], "Maximum of High-Low, High-Close, and Low-Close - represents volatility.", "Neutral")

    # Volume
    try:
        print(f"Volume: {latest_data['Volume']:.2f}")
        add_explanation('Volume', round(latest_data['Volume'],2), 
            "Volume measures the total number of shares traded during a specific period. High volume indicates strong investor interest, while Low volume may indicate weak interest.",
            "Positive" if latest_data['Volume'] > df['Volume'].mean() else "Negative")

    except:
        pass    

    # Volatility
    print(f"Volatility: {latest_data['Volatility']:.2f}")
    add_explanation('Volatility', latest_data['Volatility'], 
        "Volatility measures the dispersion of returns. High volatility suggests more risk and price fluctuations, while Low volatility suggests stability.",
        "Negative" if latest_data['Volatility'] > df['Volatility'].mean() else "Positive")

    # EMA-12 and EMA-26 Combined
    print(f"EMA-12: {latest_data['EMA_12']:.2f}")
    print(f"EMA-26: {latest_data['EMA_26']:.2f}")
    if latest_data['EMA_12'] > latest_data['EMA_26']:
        add_explanation('EMA_12_26', None, 
            "EMA-12 is above EMA-26 indicating a bullish trend, suggesting potential upward momentum.", 
            "Positive")
    else:
        add_explanation('EMA_12_26', None, 
            "EMA-12 is beLow EMA-26 indicating a bearish trend, suggesting potential downward momentum.", 
            "Negative")

    # P/E Ratio
    if latest_data['pe-ratio']:
        add_explanation('pe-ratio', latest_data['pe-ratio'], 
            "A Low P/E ratio might indicate that the stock is undervalued- Positive signal.",
            "Positive" if latest_data['pe-ratio'] < 20 
            else "Negative" if latest_data['pe-ratio'] > 30 
            else "Neutral")
        print(f"P/E Ratio: {latest_data['pe-ratio']:.2f}")
            

    return explanations


@login_required
def portfolio_view(request):
    update_market_prices()
    portfolio = Portfolio.objects.filter(user=request.user)
    total_value = sum(item.current_value for item in portfolio)
    total_profit_loss = sum(item.profit_loss for item in portfolio)
    
    asset_type_counts = {}
    for item in portfolio:
        asset_type = item.asset_type
        asset_type_counts[asset_type] = asset_type_counts.get(asset_type, 0) + item.current_value

    asset_types = list(asset_type_counts.keys())
    asset_percentages = [round((value / total_value) * 100, 2) for value in asset_type_counts.values()]

    # Example: Portfolio Performance Data
    performance_data = PortfolioPerformance.objects.order_by("date")
    performance_dates = [entry.date.strftime("%Y-%m-%d") for entry in performance_data]
    performance_values = [entry.total_value for entry in performance_data]
    
    context = {
        'portfolio': portfolio,
        'total_value': total_value,
        'total_profit_loss': total_profit_loss,
        'asset_types': asset_types,
        'asset_percentages': asset_percentages,
        'performance_dates': performance_dates,
        'performance_values': performance_values,
    }
    
    return render(request, 'home/portfolio_view.html', context)

from django.http import JsonResponse

@login_required
def portfolio_data(request):
    update_market_prices()
    update_portfolio_performance()
    hourly_portfolio_snapshot()
    portfolio = Portfolio.objects.filter(user=request.user)
    total_value = sum(item.current_value for item in portfolio)
    total_profit_loss = sum(item.profit_loss for item in portfolio)
    
    asset_type_counts = {}
    for item in portfolio:
        asset_type = item.asset_type
        asset_type_counts[asset_type] = asset_type_counts.get(asset_type, 0) + item.current_value

    asset_types = list(asset_type_counts.keys())
    asset_percentages = [round((value / total_value) * 100, 2) for value in asset_type_counts.values()]

    # Example: Portfolio Performance Data
    performance_data = PortfolioPerformance.objects.filter(user=request.user).order_by("date")
    performance_dates = [entry.date.strftime("%Y-%m-%d") for entry in performance_data]
    performance_values = [entry.total_value for entry in performance_data]
    print('*************** i found assets ***************')
    print(performance_data)
    data = {
        'portfolio': list(portfolio.values()),
        'total_value': total_value,
        'total_profit_loss': total_profit_loss,
        'asset_types': asset_types,
        'asset_percentages': asset_percentages,
        'performance_dates': performance_dates,
        'performance_values': performance_values,
    }
    
    return JsonResponse(data)


@login_required
def add_transaction(request):
    if request.method == 'POST':
        form = TransactionForm(request.POST)
        if form.is_valid():
            transaction = form.save(commit=False)
            transaction.user = request.user  # Associate with the logged-in user
            transaction.save()

            # Add success message after successful form submission
            messages.success(request, 'Transaction added successfully!')
            return redirect('portfolio_view')  # Redirect to another page (like portfolio view)
        else:
            # Add error message if form is invalid after submission
            logger.debug("Form errors: %s", form.errors) #hbar sui xrp trx avax filecoi crcoin pino
            messages.error(request, 'Please correct the errors below.')
    else:
        form = TransactionForm()

    return render(request, 'home/add_transaction.html', {'form': form})

@login_required
def transactions(request):
    print("Entered transactions view")  # Ensure this appears in the logs
    transactions = Transaction.objects.filter(user=request.user)
    print(f"Assets count: {transactions.count()}")  # Verify assets are being retrieved
    context = {
        'transactions': transactions,
        'cc': 'CC i love u',
    }
    return render(request, 'home/transactions.html', context)


@shared_task
def update_market_prices():
    # Get all portfolio items
    portfolio = Portfolio.objects.all()

    for item in portfolio:
        try:
            # Retrieve current market price from Yahoo Finance
            ticker = item.ticker
            stock = yf.Ticker(ticker)
            
            # Retrieve stock data and handle missing data
            stock_history = stock.history(period='1d')
            if stock_history.empty:
                print(f"No data found for {ticker}")
                continue  # Skip this iteration if no data is returned

            current_price = Decimal(stock_history['Close'].iloc[-1])  # Convert price to Decimal

            # Calculate the current value and ensure it's a Decimal calculation
            current_value = current_price * Decimal(item.total_quantity)

            # Debug output to verify calculations
            print(f"Updating {item.ticker}: Price = {current_price}, Quantity = {item.total_quantity}, Value = {current_value}")

            # Update the database with the new price and value
            item.current_market_price = current_price
            item.current_value = current_value  # Update current value
            item.save()  # Save changes to the database

            # Log successful update
            print(f"Successfully updated {item.ticker} with value {current_value}")

        except Exception as e:
            print(f"Error updating price for {item.ticker}: {e}")



def fetch_details(request, ticker):
    try:
        stock = yf.Ticker(ticker)
        stock_history = stock.history(period='1d')
        name = stock.info.get('shortName', 'N/A')
        asset_type = stock.info.get('quoteType', 'Unknown')  # Retrieve the asset type
        current_price = Decimal(stock_history['Close'].iloc[-1]) 
        return JsonResponse({'name': name, 'asset_type': asset_type, 'current_price': current_price})
    except Exception as e:
        return JsonResponse({'error': str(e), 'name': 'N/A', 'asset_type': 'Unknown'})
    

def fetch_chart_data(request, ticker):
    try:
        # Fetch historical data for the past year
        stock_data = yf.Ticker(ticker).history(period="1y", interval="1d")
        if stock_data.empty:
            return JsonResponse({"error": f"No data found for ticker '{ticker}'."})

        # Format the data for the chart
        dates = stock_data.index.strftime('%Y-%m-%d').tolist()
        prices = stock_data['Close'].tolist()

        return JsonResponse({"dates": dates, "prices": prices})
    except Exception as e:
        return JsonResponse({"error": f"Failed to fetch data: {str(e)}"})
    

@login_required
def asset_category_performance_data(request):
    user = request.user
    performance_data = (
        AssetCategoryPerformance.objects.filter(user=user)
        .order_by('date', 'asset_type')
    )
    data = {}
    for item in performance_data:
        if item.asset_type not in data:
            data[item.asset_type] = {"dates": [], "values": []}
        data[item.asset_type]["dates"].append(item.date.strftime('%Y-%m-%d'))
        data[item.asset_type]["values"].append(float(item.total_value))

    return JsonResponse({"performance_data": data})


    '''
    redis-server
celery -A core worker --loglevel=info
celery -A core beat --loglevel=info
'''

result = my_task.delay(3, 5)
update_portfolio=update_portfolio_performance()
update_portfolio_values=hourly_portfolio_snapshot()


from django.http import JsonResponse

def create_charts_p(request,ticker):
    period = '1y'
    interval = '1d'
    symbol = ticker
    data = fetch_data_from_yf(symbol, period, interval)
    if data is None or data.empty:
        print(f"No data available for {symbol} with period {period} and interval {interval}")
        return


    # Calculate indicators
    data = calculate_indicators(data)

    # Create traces for plotly
    traces = []

    # Closing price trace
    traces.append(go.Scatter(x=data['timestamp'], y=data['Close'], mode='lines', name='Close', line=dict(color='#00B2E2')))

    # Bollinger Bands traces
    traces.append(go.Scatter(x=data['timestamp'], y=data['BBU_20'], mode='lines', name='Bollinger Upper Band', line=dict(color='red', dash='dash')))
    traces.append(go.Scatter(x=data['timestamp'], y=data['BBL_20'], mode='lines', name='Bollinger Lower Band', line=dict(color='red', dash='dash')))

    # Moving Averages traces
    traces.append(go.Scatter(x=data['timestamp'], y=data['SMA_20'], mode='lines', name='MA 20', line=dict(color='green')))
    traces.append(go.Scatter(x=data['timestamp'], y=data['SMA_50'], mode='lines', name='MA 50', line=dict(color='orange')))
    traces.append(go.Scatter(x=data['timestamp'], y=data['SMA_200'], mode='lines', name='MA 200', line=dict(color='purple')))

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout for interactive features
    fig.update_layout(
        title=f'{symbol} - {period} {interval}',
        xaxis_title=f'Date: {data["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")}',
        yaxis_title='Value',
        template='plotly_dark',  # Dark theme
        hovermode='x unified'
    )

    # Show the plot
    fig.show()

