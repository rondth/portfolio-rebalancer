import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "log_returns",
    "volatility_lag_week",
    "volatility_lag_month",
    "volatility_lag_quarter",
    "garch_volatility",
    "absolute_returns_lag",
    "vol_change",
    "garman_klass"
]

MODEL_PATH = "xgb_model.pkl"
SCALER_PATH = "scaler.pkl"

# train model ONCE 
def train_model(tickers=None):
    """
    Train the XGBoost volatility prediction model using multiple tickers
    to ensure strong generalisation to ANY asset the user inputs.
    """

    # default training universe if none provided
    if tickers is None:
        tickers = [
            "SPY", "QQQ", "DIA", "IWM",
            "XLK", "XLF", "XLE", "XLY", "XLP",
            "TLT", "HYG",
            "GLD", "SLV", "USO", "UNG",
            "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
            "BTC-USD", "ETH-USD"
        ]

    all_rows = []   # store training rows for all tickers

    print("\n📊 Training model on multi-ticker universe...")
    print("Tickers:", tickers)

    for tkr in tickers:
        print(f"\nDownloading {tkr}...")
        try:
            df = yf.download(tkr, start="1900-01-01")
            if len(df) < 300:
                print(f"⚠ Skipping {tkr}: too little data.")
                continue
        except:
            print(f"⚠ Failed to download {tkr}, skipping.")
            continue

        # feature engineering
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1)) * 100
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
        df["target_volatility"] = df["log_returns"].rolling(window=indexer).std().shift(-1)

        df["volatility_lag_week"] = df["log_returns"].rolling(5).std().shift(1)
        df["volatility_lag_month"] = df["log_returns"].rolling(22).std().shift(1)
        df["volatility_lag_quarter"] = df["log_returns"].rolling(66).std().shift(1)
        df["absolute_returns_lag"] = abs(df["log_returns"].shift(1))

        log_hl = np.log(df["High"] / df["Low"])
        log_co = np.log(df["Close"] / df["Open"])
        df["garman_klass"] = np.sqrt(0.5 * log_hl**2 - (2*np.log(2)-1)*log_co**2) * 100

        df["vol_change"] = df["Volume"].pct_change()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # add GARCH volatility
        print(f"Fitting GARCH for {tkr}...")
        returns = df["log_returns"] - df["log_returns"].mean()
        garch = arch_model(returns, vol="EGARCH", p=1, q=1, dist="t")
        try:
            gfit = garch.fit(disp="off")
        except:
            print(f"⚠ GARCH failed for {tkr}, skipping.")
            continue

        df["garch_volatility"] = gfit.conditional_volatility

        # append only rows needed for training
        all_rows.append(df[FEATURES + ["target_volatility"]])

    if len(all_rows) == 0:
        raise Exception("❌ No data collected for training!")

    # combine all tickers into one training dataset
    full_data = pd.concat(all_rows)
    full_data.dropna(inplace=True)

    print(f"\nTotal training rows collected: {len(full_data)}")

    # train model
    X = full_data[FEATURES]
    y = full_data["target_volatility"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training XGBoost model...")
    model = XGBRegressor(
        learning_rate=0.01,
        max_depth=4,
        n_estimators=500,
        n_jobs=-1,
        objective="reg:absoluteerror"
    )

    model.fit(X_scaled, y)

    # save model + scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\n🎉 Model training complete! Saved as:")
    print(" - xgb_model.pkl")
    print(" - scaler.pkl")

    return model, scaler

# load model for prediction
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# predict next-day volatility for ANY ticker
def predict_next_day_volatility(ticker, return_prices=False):
    """
    Use the pre-trained XGBoost model + per-ticker EGARCH features
    to predict next-day volatility for ANY ticker.

    If return_prices=True, also return the Close price series
    (for use in correlation / covariance in Streamlit).
    """
    model, scaler = load_model()

    # download data
    df = yf.download(ticker, start="2015-01-01", progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")

    # handle possible MultiIndex from yfinance when using multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if 'Close' in df.columns.get_level_values(0):
                df = df.xs(ticker, axis=1, level=1, drop_level=True)
        except Exception:
            pass

    # features engineering (must match training style)
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1)) * 100

    df["volatility_lag_week"] = df["log_returns"].rolling(5).std().shift(1)
    df["volatility_lag_month"] = df["log_returns"].rolling(22).std().shift(1)
    df["volatility_lag_quarter"] = df["log_returns"].rolling(66).std().shift(1)
    df["absolute_returns_lag"] = abs(df["log_returns"].shift(1))

    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    df["garman_klass"] = np.sqrt(0.5 * log_hl**2 - (2*np.log(2)-1)*log_co**2) * 100

    df["vol_change"] = df["Volume"].pct_change()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if len(df) < 200:
        raise ValueError(f"Insufficient data for {ticker}")

    # GARCH fit on entire series for that ticker
    returns = df["log_returns"] - df["log_returns"].mean()
    garch = arch_model(returns, vol="EGARCH", p=1, q=1, dist="t")
    gfit = garch.fit(disp="off")
    df["garch_volatility"] = gfit.conditional_volatility

    # forecast next-day GARCH for the last row
    garch_forecast = gfit.forecast(horizon=1).variance.iloc[-1].values[0] ** 0.5
    df.loc[df.index[-1], "garch_volatility"] = garch_forecast

    # build feature row for LAST day
    last_row = df[FEATURES].iloc[[-1]]
    X_scaled = scaler.transform(last_row)
    pred_daily_vol = model.predict(X_scaled)[0]

    annualised_vol = pred_daily_vol * np.sqrt(252)

    if return_prices:
        return annualised_vol, df["Close"]
    else:
        return annualised_vol

