import yfinance as yf
import numpy as np
from fredapi import Fred
import pandas as pd


symbol='^GSPC'
interval='1d'
period='5d'

df = yf.download(symbol, interval=interval, period=period, progress=False)

print(df.head())

ticker = yf.Ticker(symbol)
info = ticker.info

# Get the type of the ticker
ticker_type = info.get("quoteType", "Unknown")

print(f"The type of {symbol} is: {ticker_type}")