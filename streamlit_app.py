import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Smart Portfolio Rebalancer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: 600;
        color: #ffffff;
    }
    /* Card-like containers for metrics */
    .css-1r6slb0 {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 10px;
    }
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #111827;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        height: 3em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Backend Logic (Cached) ---

@st.cache_data(show_spinner=False)
def fetch_and_predict(ticker):
    """
    Fetches data, engineers features, runs GARCH + XGBoost pipeline.
    Returns: (Predicted Annual Volatility, Historical Prices, Error Message)
    """
    try:
        # A. Data Ingestion
        data = yf.download(ticker, start="2018-01-01", progress=False)
        
        if data.empty:
            return None, None, f"No data found for {ticker}."
            
        # Clean up column structure if necessary (handle MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            # Check if 'Close' is in the top level
            try:
                if 'Close' in data.columns.get_level_values(0):
                     # Flatten if possible or select specific ticker level
                     data = data.xs(ticker, axis=1, level=1, drop_level=True)
            except:
                pass
                
        # Ensure we have the basic columns
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        if not all(col in data.columns for col in required_cols):
             # Fallback: sometimes yfinance returns multi-index differently
             if isinstance(data.columns, pd.MultiIndex):
                 data.columns = data.columns.get_level_values(0)
             
             if not all(col in data.columns for col in required_cols):
                return None, None, f"Incomplete data schema for {ticker}."

        # B. Feature Engineering
        data["log_returns"] = np.log(data["Close"] / data["Close"].shift(1)) * 100
        
        # Target: Realized volatility of the *next* 5 days
        window = 5
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
        data["target_volatility"] = data["log_returns"].rolling(window=indexer).std().shift(-1)

        # Lagged Volatility Features
        data["volatility_lag_week"] = data["log_returns"].rolling(5).std().shift(1)
        data["volatility_lag_month"] = data["log_returns"].rolling(22).std().shift(1)
        data["volatility_lag_quarter"] = data["log_returns"].rolling(66).std().shift(1)
        data["absolute_returns_lag"] = abs(data["log_returns"].shift(1))
        
        # Garman-Klass Volatility
        log_hl = np.log(data["High"] / data["Low"])
        log_co = np.log(data["Close"] / data["Open"])
        data["garman_klass"] = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2) * 100
        
        # Volume Change
        data["vol_change"] = data["Volume"].pct_change()
        
        # Cleanup
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        
        if len(data) < 252:
            return None, None, f"Insufficient data points for {ticker} (Need > 1 year)."

        # C. Modeling (GARCH + XGBoost)
        # Split Data (80/20)
        split = int(len(data) * 0.8)
        training = data.iloc[:split].copy()
        testing = data.iloc[split:].copy()
        
        # 1. GARCH (Feature Generation)
        garch_model = arch_model(training["log_returns"] - training["log_returns"].mean(), 
                                 vol="EGARCH", p=1, q=1, dist="t")
        garch_fit = garch_model.fit(disp="off")
        training["garch_volatility"] = garch_fit.conditional_volatility
        
        # Apply parameters to test set
        garch_test = arch_model(testing["log_returns"] - training["log_returns"].mean(), 
                                vol="EGARCH", p=1, q=1, dist="t")
        garch_test_fit = garch_test.fix(garch_fit.params)
        testing["garch_volatility"] = garch_test_fit.conditional_volatility
        
        # 2. XGBoost
        features = ["log_returns", "volatility_lag_week", "volatility_lag_month", 
                    "volatility_lag_quarter", "garch_volatility", "absolute_returns_lag", 
                    "vol_change", "garman_klass"]
        
        X_train = training[features]
        y_train = training["target_volatility"]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = XGBRegressor(learning_rate=0.01, max_depth=4, n_estimators=500, n_jobs=-1, objective="reg:absoluteerror")
        model.fit(X_train_scaled, y_train)
        
        # D. Final Prediction (For Tomorrow)
        # Re-fit GARCH on ALL data to get the absolute latest variance forecast
        full_data = pd.concat([training, testing])
        full_garch = arch_model(full_data["log_returns"] - full_data["log_returns"].mean(), vol="EGARCH", p=1, q=1, dist="t")
        full_garch_fit = full_garch.fix(garch_fit.params)
        
        # Forecast 1 step ahead
        next_day_variance = full_garch_fit.forecast(horizon=1).variance.iloc[-1].values[0]
        next_day_garch_vol = np.sqrt(next_day_variance)
        
        # Prepare input vector for XGBoost
        last_row = full_data.iloc[[-1]][features].copy()
        last_row["garch_volatility"] = next_day_garch_vol # Update with forecast
        
        last_row_scaled = scaler.transform(last_row)
        predicted_vol = model.predict(last_row_scaled)[0]
        
        annualized_vol = predicted_vol * np.sqrt(252)
        
        return annualized_vol, full_data["Close"], None

    except Exception as e:
        return None, None, str(e)

# --- 3. UI Layout ---

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=60)
    st.title("Smart Rebalancer")
    st.caption("Group 1: Dexun, Thaddus, Eron")
    st.divider()
    
    st.markdown("### ⚙️ Settings")
    input_tickers = st.text_area("Assets (Comma Separated)", "AAPL, NVDA, MSFT, BTC-USD, GLD", help="Enter stock or crypto symbols").upper()
    risk_free_rate = st.number_input("Risk Free Rate (%)", value=4.0, step=0.1) / 100
    mc_sims = st.slider("Monte Carlo Simulations", 500, 5000, 1000)
    
    st.divider()
    st.markdown("Developed with **Streamlit** & **XGBoost**")

# Main Tabs
tab_home, tab_how, tab_app = st.tabs(["🏠 Home / Proposal", "🧠 How It Works", "🚀 Live Optimizer"])

# --- TAB 1: HOME ---
with tab_home:
    st.title("ML-Powered Smart Portfolio Rebalancer")
    st.markdown("*A Capstone Project by Group 1*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📌 Problem Statement")
        st.write("""
        Modern investors struggle to maintain balanced portfolios in volatile markets. 
        Traditional tools provide static allocations, often missing opportunities to optimize returns or control risks dynamically. 
        Most retail investors lack tools to:
        * Avoid emotional rebalancing.
        * Detect overexposure to specific assets.
        * Forecast short-term risk accurately.
        """)
        
        st.subheader("🎯 Project Objective")
        st.write("""
        Build a machine learning-powered system that:
        1. **Predicts** short-term volatility using XGBoost & GARCH.
        2. **Evaluates** portfolio risk in real-time.
        3. **Recommends** optimal rebalancing actions (Buy/Sell) to maximize the Sharpe Ratio.
        """)

    with col2:
        st.info("### Key Features\n"
                "✅ **Multi-Asset Support** (Stocks & Crypto)\n\n"
                "✅ **Hybrid ML Models** (GARCH + XGBoost)\n\n"
                "✅ **Monte Carlo** Simulation\n\n"
                "✅ **Interactive** Dashboard")
        
        st.image("https://images.unsplash.com/photo-1611974765270-ca1258634369?q=80&w=1000&auto=format&fit=crop", caption="Algorithmic Trading", use_container_width=True)

# --- TAB 2: HOW IT WORKS ---
with tab_how:
    st.header("The Architecture")
    st.write("Our system pipelines financial data through three distinct stages to generate actionable insights.")
    
    st.markdown("### 1. Feature Engineering")
    st.code("""
    # Inputs:
    - Log Returns
    - Garman-Klass Volatility (High/Low/Open/Close)
    - Volume Shocks
    - Lagged Volatility (Week, Month, Quarter)
    """, language="python")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 2. The Hybrid Model")
        st.write("""
        We use a **Residual Learning** approach:
        1. **GARCH (EGARCH)** captures the 'clustering' of volatility (heteroskedasticity).
        2. **XGBoost** takes the GARCH output + other market features to correct the error and handle non-linear relationships.
        """)
    with col_b:
        st.markdown("### 3. Optimization Engine")
        st.write("""
        We don't just predict risk; we use it.
        1. **Prediction:** ML model outputs `Predicted Volatility` for tomorrow.
        2. **Covariance:** We construct a forward-looking covariance matrix.
        3. **Markowitz:** We run Monte Carlo simulations to find the 'Efficient Frontier'.
        """)

# --- TAB 3: LIVE APP ---
with tab_app:
    st.header("Portfolio Dashboard")
    
    # Process Inputs
    tickers = [t.strip() for t in input_tickers.split(",") if t.strip()]
    
    st.subheader("Step 1: Define Current Portfolio")
    
    # Create an initial dataframe for user input
    if 'current_weights_df' not in st.session_state:
        # Default equal weight for initial load
        st.session_state.current_weights_df = pd.DataFrame({
            "Asset": tickers,
            "Current Weight (%)": [round(100.0/len(tickers), 2)] * len(tickers)
        })
    else:
        # Update if ticker list changed length (basic sync)
        if len(st.session_state.current_weights_df) != len(tickers):
             st.session_state.current_weights_df = pd.DataFrame({
                "Asset": tickers,
                "Current Weight (%)": [round(100.0/len(tickers), 2)] * len(tickers)
            })
        else:
             # Ensure assets match
             st.session_state.current_weights_df["Asset"] = tickers

    # Interactive Data Editor
    edited_df = st.data_editor(
        st.session_state.current_weights_df,
        column_config={
            "Current Weight (%)": st.column_config.NumberColumn(
                "Current Weight (%)",
                help="Enter your current portfolio percentage (0-100)",
                min_value=0,
                max_value=100,
                step=0.1,
                format="%.1f%%"
            )
        },
        disabled=["Asset"],
        hide_index=True,
        use_container_width=True
    )
    
    total_weight = edited_df["Current Weight (%)"].sum()
    if not (99.0 <= total_weight <= 101.0):
        st.warning(f"⚠️ Current weights sum to {total_weight:.1f}%. They should sum to roughly 100%.")

    st.subheader("Step 2: Run Optimization")
    
    if st.button("🚀 Run Analysis & Rebalance", type="primary"):
        if len(tickers) < 2:
            st.error("Please enter at least 2 assets to create a portfolio.")
        else:
            # 1. Calculation Phase
            status = st.status("Processing Market Data...", expanded=True)
            
            results = {}
            prices_df = pd.DataFrame()
            failed = []
            
            progress_bar = status.progress(0)
            
            for i, t in enumerate(tickers):
                status.write(f"Analyzing **{t}** with XGBoost...")
                vol, price_series, err = fetch_and_predict(t)
                
                if err:
                    failed.append(f"{t}: {err}")
                else:
                    results[t] = vol
                    # Align price series dates
                    if prices_df.empty:
                        prices_df = pd.DataFrame(price_series)
                        prices_df.columns = [t]
                    else:
                        prices_df = prices_df.join(price_series.rename(t), how="inner")
                
                progress_bar.progress((i + 1) / len(tickers))
            
            if failed:
                for f in failed:
                    status.warning(f)
            
            valid_assets = list(results.keys())
            
            if len(valid_assets) < 2:
                status.error("Not enough valid assets to optimize. Check your tickers.")
                status.update(state="error")
            else:
                status.write("Running Monte Carlo Simulations...")
                
                # 2. Optimization Phase
                returns_log = np.log(prices_df / prices_df.shift(1)).dropna()
                corr_matrix = returns_log.corr()
                
                # D = Diagonal matrix of Predicted Annual Volatilities
                vols_vector = np.array([results[a] for a in valid_assets]) / 100 
                D = np.diag(vols_vector)
                
                future_cov = D @ corr_matrix.values @ D
                
                # Monte Carlo
                sim_results = []
                mean_daily_ret = returns_log.mean()
                mean_annual_ret = mean_daily_ret * 252
                
                for _ in range(mc_sims):
                    w = np.random.random(len(valid_assets))
                    w /= np.sum(w)
                    
                    p_ret = np.dot(w, mean_annual_ret[valid_assets])
                    p_var = w.T @ future_cov @ w
                    p_vol = np.sqrt(p_var)
                    p_sharpe = (p_ret - risk_free_rate) / p_vol
                    
                    sim_results.append([p_ret, p_vol, p_sharpe] + list(w))
                
                columns = ['Return', 'Volatility', 'Sharpe'] + valid_assets
                df_sim = pd.DataFrame(sim_results, columns=columns)
                
                # Best Portfolios
                max_sharpe_port = df_sim.iloc[df_sim['Sharpe'].idxmax()]
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                # 3. Visualization Phase
                
                # Row 1: Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Optimal Sharpe Ratio", f"{max_sharpe_port['Sharpe']:.2f}", delta="Maximized")
                m2.metric("Predicted Annual Return", f"{max_sharpe_port['Return']:.2%}")
                m3.metric("Predicted Annual Risk", f"{max_sharpe_port['Volatility']:.2%}", delta_color="inverse")
                
                st.divider()
                
                # Row 2: Charts
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.subheader("Efficient Frontier")
                    fig_ef = px.scatter(
                        df_sim, x="Volatility", y="Return", color="Sharpe",
                        title="Monte Carlo Simulation (ML-Adjusted)",
                        color_continuous_scale="RdYlGn",
                        labels={"Return": "Expected Return", "Volatility": "Predicted Volatility"}
                    )
                    # Add Star for Max Sharpe
                    fig_ef.add_trace(go.Scatter(
                        x=[max_sharpe_port['Volatility']], 
                        y=[max_sharpe_port['Return']],
                        mode='markers', marker=dict(color='red', size=15, symbol='star'),
                        name="Optimal Portfolio"
                    ))
                    st.plotly_chart(fig_ef, use_container_width=True)
                
                with c2:
                    st.subheader("Current vs Optimal")
                    
                    # Prepare comparison data
                    # Get user weights, normalize to match valid assets if some failed
                    user_weights_map = {row['Asset']: row['Current Weight (%)']/100.0 for _, row in edited_df.iterrows()}
                    
                    comp_data = []
                    for asset in valid_assets:
                        curr = user_weights_map.get(asset, 0.0)
                        opt = max_sharpe_port[asset]
                        comp_data.append({"Asset": asset, "Type": "Current", "Weight": curr})
                        comp_data.append({"Asset": asset, "Type": "Optimal", "Weight": opt})
                    
                    df_comp = pd.DataFrame(comp_data)
                    
                    fig_comp = px.bar(df_comp, x="Asset", y="Weight", color="Type", barmode="group",
                                      color_discrete_map={"Current": "#94a3b8", "Optimal": "#22c55e"})
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                # Row 3: Actionable Table
                st.subheader("📋 Rebalancing Recommendations")
                
                rebal_data = []
                for asset in valid_assets:
                    curr_w = user_weights_map.get(asset, 0.0)
                    targ_w = max_sharpe_port[asset]
                    diff = targ_w - curr_w
                    
                    action = "HOLD"
                    if diff > 0.01: action = f"BUY (+{diff:.1%})"
                    elif diff < -0.01: action = f"SELL ({diff:.1%})"
                    
                    rebal_data.append({
                        "Asset": asset,
                        "Current Weight": f"{curr_w:.1%}",
                        "Target Weight": f"{targ_w:.1%}",
                        "Difference": f"{diff:+.1%}",
                        "Action": action,
                        "Predicted Vol": f"{results[asset]:.2f}%"
                    })
                
                st.dataframe(pd.DataFrame(rebal_data), hide_index=True, use_container_width=True)
                
                st.info("💡 **Note:** 'Target Weight' is derived from maximizing the Sharpe Ratio using XGBoost volatility predictions for tomorrow.")

    else:
        st.write("👈 **Adjust settings in the sidebar**, define your current weights above, and click 'Run Analysis'.")