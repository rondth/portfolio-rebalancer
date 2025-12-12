# ML Smart Portfolio Rebalancer
**Live App:** https://nusfintech-portfolio-optimiser.streamlit.app/

A machine-learning powered portfolio optimization tool that predicts volatility and generates optimal asset allocation using EGARCH, XGBoost, and Monte Carlo simulation.

## Features
- Volatility Prediction: EGARCH + XGBoost hybrid model
- Smart Returns: Blended arithmetic & geometric mean
- Portfolio Optimization: Monte Carlo simulation with Sharpe ratio maximization
- Interactive Dashboard: Streamlit web interface
- Rebalancing: Automated Buy/Sell/Hold recommendations

## Quick Start
Installation
```bash
pip install -r requirements.txt
```
Run the App
```bash
streamlit run streamlit_app.py
```

## Usage
1. Enter tickers: AAPL, NVDA, MSFT, BTC-USD
2. Set current portfolio weights
3. Click Run Optimization
4. View results and rebalancing plan

## How It Works
### Volatility Prediction
- EGARCH(1,1): Captures volatility clustering
- XGBoost: Uses engineered features (returns, volume, lags)
- Output: Next-day annualized volatility

## Optimisation
Monte Carlo simulation → Maximum Sharpe ratio → Optimal weights

## Expected Returns
$$R = 0.5 \cdot R_{\text{arith}} + 0.5 \cdot R_{\text{geo}}$$

## Covariance Matrix
$$\Sigma_{\text{future}} = D \cdot Corr \cdot D$$

## Authors
- Dexun
- Thaddus
- Eron
