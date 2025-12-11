import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="ML Smart Portfolio Rebalancer",
    layout="wide",
)

# --- 2. The Core Algorithm ---

@st.cache_data(show_spinner=False)
def predict_next_day_volatility(ticker):
    """
    Replicates the exact logic from 'fintech_capstone_project.py'.
    """
    try:
        # 1. Data Loading
        data = yf.download(ticker, start="2015-01-01", progress=False)
        
        if data.empty:
            return None, None, f"No data for {ticker}"

        # Handle yfinance MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            try:
                if 'Close' in data.columns.get_level_values(0):
                    data = data.xs(ticker, axis=1, level=1, drop_level=True)
            except:
                pass
        
        # Ensure column names are formatted
        data.columns = [c.capitalize() for c in data.columns]
        
        # 2. Feature Engineering
        data["log_returns"] = np.log(data["Close"] / data["Close"].shift(1)) * 100

        window = 5
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
        data["target_volatility"] = data["log_returns"].rolling(window=indexer).std().shift(-1)

        data["volatility_lag_week"] = data["log_returns"].rolling(5).std().shift(1)
        data["volatility_lag_month"] = data["log_returns"].rolling(22).std().shift(1)
        data["volatility_lag_quarter"] = data["log_returns"].rolling(66).std().shift(1)

        data["absolute_returns_lag"] = abs(data["log_returns"].shift(1))
        
        # Garman Klass Formula
        log_hl = np.log(data["High"] / data["Low"])
        log_co = np.log(data["Close"] / data["Open"])
        data["garman_klass"] = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2) * 100

        data["vol_change"] = data["Volume"].pct_change()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        if len(data) < 200:
            return None, None, "Insufficient data"

        # 3. GARCH Modeling
        split_idx = int(len(data)*0.8)
        training = data.iloc[:split_idx].copy()
        testing = data.iloc[split_idx:].copy()

        training_garch_model = arch_model(training["log_returns"] - training["log_returns"].mean(), vol="EGARCH", p=1, q=1, dist="t")
        training_garch_model_fit = training_garch_model.fit(disp="off")
        training["garch_volatility"] = training_garch_model_fit.conditional_volatility

        testing_garch_model = arch_model(testing["log_returns"] - training["log_returns"].mean(), vol="EGARCH", p=1, q=1, dist="t")
        testing_garch_model_fix = testing_garch_model.fix(training_garch_model_fit.params)
        testing["garch_volatility"] = testing_garch_model_fix.conditional_volatility

        # 4. XGBoost Modeling
        features = ["log_returns", "volatility_lag_week", "volatility_lag_month", "volatility_lag_quarter", "garch_volatility", "absolute_returns_lag", "vol_change", "garman_klass"]

        X_train = training[features]
        y_train = training["target_volatility"]
        X_test = testing[features]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = XGBRegressor(learning_rate=0.01, max_depth=4, n_estimators=500, n_jobs=-1, objective="reg:absoluteerror")
        
        val_split = int(len(X_train) * 0.9)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # 5. Final Prediction
        data_full = pd.concat([training, testing], axis=0)
        last_row = data_full.iloc[[-1]][features].copy()
        
        full_garch = arch_model(data_full["log_returns"] - data_full["log_returns"].mean(), vol="EGARCH", p=1, q=1, dist="t")
        full_garch_fit = full_garch.fix(training_garch_model_fit.params)
        next_day_garch = full_garch_fit.forecast(horizon=1).variance.iloc[-1].values[0]**0.5
        
        last_row["garch_volatility"] = next_day_garch
        last_row = scaler.transform(last_row)

        predicted_daily_vol = model.predict(last_row)[0]
        annualised_vol = predicted_daily_vol * np.sqrt(252)

        return annualised_vol, data_full["Close"], None

    except Exception as e:
        return None, None, str(e)

# --- 3. UI Implementation ---

logo_col1, logo_col2, logo_col3 = st.sidebar.columns([1, 2, 1])
with logo_col2:
    st.image("https://img.icons8.com/fluency/96/bullish.png", use_container_width=True)

st.sidebar.title("Smart Rebalancer")
st.sidebar.caption("Group 1: Dexun, Thaddus, Eron")
st.sidebar.divider()

# Inputs
default_tickers = "AAPL, NVDA, MSFT, BTC-USD, ETH-USD"
ticker_input = st.sidebar.text_area("Stocks (Comma Separated)", value=default_tickers)
risk_free_rate = st.sidebar.number_input("Risk Free Rate", value=0.04, step=0.01)
num_simulations = st.sidebar.slider("Monte Carlo Simulations", min_value=1000, max_value=10000, value=2500, step=500, help="Higher simulations = more accurate but slower.")

st.sidebar.divider()
st.sidebar.info("Navigate tabs above to switch between the Proposal details and the Live Tool.")

# --- TABS CONFIGURATION ---
tab_info, tab_app = st.tabs(["Proposal & Methodology", "Live Optimizer"])

# --- TAB 1: PROPOSAL & METHODOLOGY ---
with tab_info:
    # --- SECTION 1: PROPOSAL ---
    st.title("ML-Powered Smart Portfolio Rebalancer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Problem Statement")
        st.write("""
        Modern investors struggle to maintain balanced portfolios in volatile markets. 
        Traditional tools provide static allocations, often missing opportunities to optimize returns or control risks dynamically. 
        Most retail investors lack tools to:
        * Avoid emotional rebalancing.
        * Detect overexposure to specific assets.
        * Forecast short-term risk accurately.
        """)
        
        st.subheader("Project Objective")
        st.write("""
        Build a machine learning-powered system that:
        1. **Predicts** short-term volatility using XGBoost & EGARCH.
        2. **Evaluates** portfolio risk in real-time.
        3. **Recommends** optimal rebalancing actions (Buy/Sell) to maximize the Sharpe Ratio.
        """)

    with col2:
        st.info("### Key Features\n"
                "• **Multi-Asset Support** (Stocks & Crypto)\n\n"
                "• **Hybrid ML Models** (EGARCH + XGBoost)\n\n"
                "• **Monte Carlo** Simulation\n\n"
                "• **Interactive** Dashboard")
    
    # --- SECTION 2: HOW IT WORKS ---
    st.divider()
    st.header("How It Works")
    st.write("Our system pipelines financial data through three distinct stages to generate actionable insights.")
    
    # Architecture Columns
    c_a, c_b, c_c = st.columns(3)
    
    with c_a:
        st.markdown("### 1. Feature Engineering")
        st.write("We extract advanced statistical features from raw market data.")
        st.code("""
# Inputs:
- Log Returns
- Garman-Klass Volatility
- Volume Shocks
- Lagged Volatility
        """, language="python")

    with c_b:
        st.markdown("### 2. The Hybrid Model")
        st.write("We use a **Residual Learning** approach:")
        st.info("""
        1. **EGARCH (a variant of GARCH)** captures volatility clustering.
        2. **XGBoost** corrects the error using non-linear market features.
        """)

    with c_c:
        st.markdown("### 3. Optimization")
        st.write("We turn predictions into decisions:")
        st.success("""
        1. Predict **Next-Day Volatility**.
        2. Construct **Future Covariance Matrix**.
        3. Run **Monte Carlo** to maximize Sharpe Ratio.
        """)

# --- TAB 2: LIVE APP ---
with tab_app:
    st.title("Live Portfolio Dashboard")
    
    # Process Tickers
    stocks = [s.strip().upper() for s in ticker_input.split(",") if s.strip()]

    # Initialize Session State for Weights if needed
    if "weights_df" not in st.session_state:
        # Default equal weight
        eq_weight = round(100.0 / len(stocks), 2) if len(stocks) > 0 else 0
        st.session_state.weights_df = pd.DataFrame({"Asset": stocks, "Current Weight (%)": [eq_weight]*len(stocks)})
    else:
        # Sync if ticker list changes length
        if len(st.session_state.weights_df) != len(stocks):
            eq_weight = round(100.0 / len(stocks), 2) if len(stocks) > 0 else 0
            st.session_state.weights_df = pd.DataFrame({"Asset": stocks, "Current Weight (%)": [eq_weight]*len(stocks)})
        else:
            # FIX: Ensure we update the 'Asset' column, not create a new 'Ticker' column
            st.session_state.weights_df["Asset"] = stocks

    col_input, col_check = st.columns([2, 1])
    
    with col_input:
        st.subheader("1. Define Current Allocation")
        st.write("Enter your **Current Portfolio Weights**:")
        edited_df = st.data_editor(
            st.session_state.weights_df, 
            column_config={
                "Current Weight (%)": st.column_config.NumberColumn(
                    "Current Weight (%)",
                    min_value=0,
                    max_value=100,
                    step=0.1,
                    format="%.1f%%"
                )
            },
            use_container_width=True, 
            hide_index=True
        )

    # --- 100% VALIDATION LOGIC ---
    total_weight = edited_df["Current Weight (%)"].sum()
    is_valid_sum = 99.0 <= total_weight <= 101.0
    
    with col_check:
        st.subheader("Validation")
        st.metric("Total Allocation", f"{total_weight:.1f}%")
        
        if is_valid_sum:
            st.success("Allocation is valid.")
            ready_to_run = True
        else:
            st.error("Weights must sum to approx 100%.")
            ready_to_run = False

    st.divider()
    
    # Run Button
    if st.button("Run Prediction & Optimization", type="primary", disabled=not ready_to_run):
        if len(stocks) < 2:
            st.error("Please enter at least 2 stocks.")
        else:
            # --- EXECUTION PHASE ---
            status = st.status("Processing Models...", expanded=True)
            
            # 1. Prediction Loop
            predicted_annual_volatilities = []
            valid_stocks = []
            price_data_list = []
            
            progress_bar = status.progress(0)
            
            for i, stock in enumerate(stocks):
                status.write(f"Analyzing **{stock}** (EGARCH + XGBoost)...")
                vol, prices, err = predict_next_day_volatility(stock)
                
                if vol is not None:
                    predicted_annual_volatilities.append(vol)
                    valid_stocks.append(stock)
                    price_data_list.append(prices.rename(stock))
                else:
                    status.warning(f"Failed to process {stock}: {err}")
                
                progress_bar.progress((i + 1) / len(stocks))
            
            if len(valid_stocks) < 2:
                status.update(label="Error: Not enough valid data", state="error")
                st.error("Not enough valid stocks to optimize.")
            else:
                # 2. Covariance Matrix Calculation
                status.write("Constructing Future Covariance Matrix...")
                price_df = pd.concat(price_data_list, axis=1)
                returns_data = np.log(price_df / price_df.shift(1))
                returns_data.dropna(inplace=True)
                
                correlation_matrix = returns_data.corr()
                
                # The Future Covariance
                D = np.diag(np.array(predicted_annual_volatilities) / 100)
                future_covariance_matrix = D @ correlation_matrix.values @ D

                # --- MODEL OUTPUTS ---
                status.write("Generating Intermediate Outputs...")
                st.divider()
                st.subheader("Intermediate Model Outputs")
                col_mod1, col_mod2 = st.columns(2)
                with col_mod1:
                    st.markdown("**Predicted Annualized Volatilities (XGBoost):**")
                    vol_df = pd.DataFrame({
                        "Asset": valid_stocks, 
                        "Predicted Vol (%)": [f"{v:.4f}" for v in predicted_annual_volatilities]
                    })
                    st.dataframe(vol_df, hide_index=True)
                with col_mod2:
                    st.markdown("**Future Covariance Matrix (Annualized):**")
                    st.dataframe(pd.DataFrame(future_covariance_matrix, columns=valid_stocks, index=valid_stocks))
                st.divider()
                # ---------------------

                # 3. Monte Carlo Simulation
                status.write(f"Running {num_simulations} Monte Carlo Simulations...")
                num_portfolios = num_simulations
                
                returns = []
                volatilities = []
                allocations = []
                
                mean_daily_returns = returns_data.mean()
                mean_annual_returns = mean_daily_returns * 252
                mean_annual_returns = mean_annual_returns[valid_stocks]

                for port in range(num_portfolios):
                    w = np.random.random(len(valid_stocks))
                    w /= np.sum(w)
                    allocations.append(w)
                    
                    returns.append(np.dot(w, mean_annual_returns))
                    
                    var = np.transpose(w).dot(future_covariance_matrix).dot(w)
                    volatilities.append(np.sqrt(var))

                # Create Portfolio DataFrame
                data2 = {"Returns": returns, "Volatilities": volatilities}
                for counter, symbol in enumerate(valid_stocks):
                    data2[symbol] = [allocation[counter] for allocation in allocations]
                
                portfolio = pd.DataFrame(data2)
                
                sharpe_ratios = (portfolio["Returns"] - risk_free_rate) / portfolio["Volatilities"]
                max_sharpe_idx = sharpe_ratios.idxmax()
                highest_sharpe_portfolio = portfolio.iloc[max_sharpe_idx]

                status.update(label="Optimization Complete", state="complete", expanded=False)

                # --- 4. Final Display ---
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Optimal Sharpe Ratio", f"{sharpe_ratios.max():.2f}", delta="Maximized")
                m2.metric("Expected Return", f"{highest_sharpe_portfolio['Returns']*100:.2f}%")
                m3.metric("Expected Volatility", f"{highest_sharpe_portfolio['Volatilities']*100:.2f}%", delta_color="inverse")
                
                st.divider()
                
                # Charts
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.subheader("Efficient Frontier")
                    fig = px.scatter(
                        portfolio, 
                        x="Volatilities", 
                        y="Returns", 
                        title="Monte Carlo Simulation (ML-Adjusted)",
                        labels={"Volatilities": "Expected Volatility", "Returns": "Expected Returns"},
                        color=sharpe_ratios,
                        color_continuous_scale="Viridis"
                    )
                    fig.add_trace(go.Scatter(
                        x=[highest_sharpe_portfolio["Volatilities"]],
                        y=[highest_sharpe_portfolio["Returns"]],
                        mode='markers',
                        marker=dict(color='red', size=20, symbol='star'),
                        name='Max Sharpe Ratio'
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    st.subheader("Comparison")
                    # Prepare comparison data
                    user_input_map = dict(zip(edited_df["Asset"], edited_df["Current Weight (%)"]))
                    
                    comp_data = []
                    for asset in valid_stocks:
                        curr = user_input_map.get(asset, 0.0)
                        opt = highest_sharpe_portfolio[asset] * 100
                        comp_data.append({"Asset": asset, "Type": "Current", "Weight": curr})
                        comp_data.append({"Asset": asset, "Type": "Optimal", "Weight": opt})
                    
                    df_comp = pd.DataFrame(comp_data)
                    
                    fig_comp = px.bar(df_comp, x="Asset", y="Weight", color="Type", barmode="group",
                                      color_discrete_map={"Current": "#94a3b8", "Optimal": "#22c55e"})
                    st.plotly_chart(fig_comp, use_container_width=True)

                # --- 5. Rebalancing Recommendations ---
                st.subheader("Rebalancing Action Plan")
                
                rebal_rows = []
                for stock in valid_stocks:
                    optimal_weight = highest_sharpe_portfolio[stock] * 100
                    current_weight = user_input_map.get(stock, 0.0)
                    diff = optimal_weight - current_weight
                    
                    action = "HOLD"
                    if diff > 1.0: action = f"BUY (+{diff:.1f}%)"
                    elif diff < -1.0: action = f"SELL ({diff:.1f}%)"
                    
                    rebal_rows.append({
                        "Asset": stock,
                        "Current Weight": f"{current_weight:.2f}%",
                        "Optimal Weight": f"{optimal_weight:.2f}%",
                        "Difference": f"{diff:+.2f}%",
                        "Action": action,
                        "Predicted Vol": f"{predicted_annual_volatilities[valid_stocks.index(stock)]:.2f}%"
                    })
                
                df_rebal = pd.DataFrame(rebal_rows)
                df_rebal.index = df_rebal.index + 1
                st.dataframe(df_rebal, use_container_width=True)