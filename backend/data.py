import yfinance as yf
import numpy as np
import pandas as pd


def download_prices(stocks, start="1900-01-01"):
    prices = yf.download(stocks, start=start)["Close"]
    prices = prices.dropna()
    return prices


def compute_log_returns(price_df):
    returns = np.log(price_df / price_df.shift(1))
    returns.dropna(inplace=True)
    return returns


def historical_correlation(stocks):
    prices = download_prices(stocks)
    returns = compute_log_returns(prices)
    corr = returns.corr()
    return corr.loc[stocks, stocks], returns
