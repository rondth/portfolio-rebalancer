from backend.model import predict_next_day_volatility
from backend.data import historical_correlation
from backend.allocator import (
    future_covariance_matrix,
    simulate_portfolios,
    max_sharpe
)

def main():
    # user inputs tickers
    stocks = []
    print("Enter tickers (Q to quit):")

    while True:
        x = input("> ")
        if x.lower() == "q":
            break
        stocks.append(x.upper())

    if not stocks:
        print("No tickers entered.")
        return

    # predict next-day vols
    predicted_vols = []
    print("\nPredicting next-day volatility...\n")
    for s in stocks:
        vol = predict_next_day_volatility(s)
        predicted_vols.append(vol)
        print(f"{s}: {vol:.4f}% annualized")

    # 3. historical correlation
    corr_matrix, returns_data = historical_correlation(stocks)
    print("\nHistorical Correlation Matrix:")
    print(corr_matrix)

    # future covariance
    fut_cov = future_covariance_matrix(predicted_vols, corr_matrix)
    print("\nFuture Covariance Matrix:")
    print(fut_cov)

    # Monte Carlo portfolios
    print("\nRunning Monte Carlo simulations...")
    portfolio_df = simulate_portfolios(stocks, fut_cov, returns_data)

    # max sharpe portfolio
    best = max_sharpe(portfolio_df)
    print("\nHighest Sharpe Ratio Portfolio:")
    print(best)


if __name__ == "__main__":
    main()
