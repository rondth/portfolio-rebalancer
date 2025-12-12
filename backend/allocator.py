import numpy as np
import pandas as pd


def future_covariance_matrix(predicted_vols, correlation_matrix):
    D = np.diag(np.array(predicted_vols) / 100)
    cov = D @ correlation_matrix @ D
    return cov


def simulate_portfolios(stocks, future_cov, returns_data, num_portfolios=1000):
    num_stocks = len(stocks)
    mean_daily = returns_data.mean()
    mean_annual = mean_daily * 252
    mean_annual = mean_annual[stocks]

    all_returns = []
    all_vols = []
    all_allocs = []

    for _ in range(num_portfolios):
        w = np.random.random(num_stocks)
        w /= np.sum(w)

        all_allocs.append(w)
        all_returns.append(np.dot(w, mean_annual))

        variance = w.T @ future_cov @ w
        all_vols.append(np.sqrt(variance))

    df = {
        "Returns": all_returns,
        "Volatilities": all_vols
    }

    for idx, sym in enumerate(stocks):
        df[f"{sym} Weight"] = [alloc[idx] for alloc in all_allocs]

    return pd.DataFrame(df)


def max_sharpe(portfolio_df, risk_free_rate=0.04):
    sharpe = (portfolio_df["Returns"] - risk_free_rate) / portfolio_df["Volatilities"]
    idx = sharpe.idxmax()
    return portfolio_df.iloc[idx]
