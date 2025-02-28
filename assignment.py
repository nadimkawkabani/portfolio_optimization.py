import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yfinance as yf
import quantstats as qs
import cvxpy as cp

# Configure Streamlit page
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

# Define available assets
AVAILABLE_ASSETS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META",    # Technology
    "NFLX", "ADSK", "INTC", "NVDA", "TSM",      # Semiconductors & Tech
    "JPM", "GS", "BAC", "C", "AXP",            # Finance
    "UNH", "LLY", "PFE", "BMY", "MRK"          # Healthcare
]

# Sidebar filters
st.sidebar.header("Filter Options")
ASSETS = st.sidebar.multiselect("Select Assets", AVAILABLE_ASSETS, default=AVAILABLE_ASSETS[:5])
date_range = st.sidebar.date_input("Select Date Range", [pd.to_datetime("2017-01-01"), pd.to_datetime("2023-12-31")])
gamma_range = st.sidebar.slider("Select Gamma Range (Risk Aversion)", min_value=-3.0, max_value=3.0, value=(0.0, 2.0))

# Fetch stock data
@st.cache_data
def get_data(assets, start, end):
    return yf.download(assets, start=start, end=end)["Adj Close"]

# Portfolio Optimization Function
def optimize_portfolio(returns, gamma_values):
    avg_returns = returns.mean().values
    cov_mat = returns.cov().values
    n_assets = len(returns.columns)
    
    weights = cp.Variable(n_assets)
    gamma_par = cp.Parameter(nonneg=True)
    portf_rtn_cvx = avg_returns @ weights
    portf_vol_cvx = cp.quad_form(weights, cov_mat)
    
    problem = cp.Problem(cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx), 
                         [cp.sum(weights) == 1, weights >= 0])
    
    portf_rtn_cvx_ef, portf_vol_cvx_ef = [], []
    
    for gamma in np.logspace(gamma_values[0], gamma_values[1], num=25):
        gamma_par.value = gamma
        problem.solve()
        portf_vol_cvx_ef.append(np.sqrt(portf_vol_cvx.value))
        portf_rtn_cvx_ef.append(portf_rtn_cvx.value)
    
    return portf_rtn_cvx_ef, portf_vol_cvx_ef

# Efficient Frontier Plot
def plot_efficient_frontier():
    prices_df = get_data(ASSETS, date_range[0], date_range[1])
    returns = prices_df.pct_change().dropna()

    portf_rtn, portf_vol = optimize_portfolio(returns, gamma_range)

    fig, ax = plt.subplots()
    ax.plot(portf_vol, portf_rtn, "g-", linewidth=2)

    # Add individual assets to the plot
    avg_returns = returns.mean().values
    cov_mat = returns.cov().values
    MARKERS = ["o", "X", "d", "*"]

    for i, asset in enumerate(returns.columns):
        ax.scatter(np.sqrt(cov_mat[i, i]), avg_returns[i], marker=MARKERS[i % len(MARKERS)], label=asset, s=100)
    
    ax.set(title="Efficient Frontier", xlabel="Volatility", ylabel="Expected Returns")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)

# Leverage Frontier Plot
def plot_leverage_frontier():
    prices_df = get_data(ASSETS, date_range[0], date_range[1])
    returns = prices_df.pct_change().dropna()

    avg_returns = returns.mean().values
    cov_mat = returns.cov().values
    n_assets = len(returns.columns)
    
    weights = cp.Variable(n_assets)
    gamma_par = cp.Parameter(nonneg=True)
    portf_rtn_cvx = avg_returns @ weights
    portf_vol_cvx = cp.quad_form(weights, cov_mat)
    
    problem = cp.Problem(cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx), 
                         [cp.sum(weights) == 1, weights >= 0])
    
    LEVERAGE_RANGE = [1, 2, 5]
    N_POINTS = 25
    portf_vol_l = np.zeros((N_POINTS, len(LEVERAGE_RANGE)))
    portf_rtn_l = np.zeros((N_POINTS, len(LEVERAGE_RANGE)))

    for lev_ind, leverage in enumerate(LEVERAGE_RANGE):
        for gamma_ind, gamma in enumerate(np.logspace(gamma_range[0], gamma_range[1], num=N_POINTS)):
            max_leverage = cp.Parameter(value=leverage)
            prob_with_leverage = cp.Problem(cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx),
                                            [cp.sum(weights) == 1, cp.norm(weights, 1) <= max_leverage])
            gamma_par.value = gamma
            prob_with_leverage.solve()
            portf_vol_l[gamma_ind, lev_ind] = np.sqrt(portf_vol_cvx.value)
            portf_rtn_l[gamma_ind, lev_ind] = portf_rtn_cvx.value
    
    # Plot
    fig, ax = plt.subplots()
    for i, leverage in enumerate(LEVERAGE_RANGE):
        ax.plot(portf_vol_l[:, i], portf_rtn_l[:, i], label=f"Leverage {leverage}")

    ax.set(title="Efficient Frontier for Different Max Leverage", xlabel="Volatility", ylabel="Expected Returns")
    ax.legend(title="Max Leverage")
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit UI
st.title("Portfolio Optimization and Efficient Frontier")
st.sidebar.markdown("### Configure Your Portfolio")

# Display options
analysis_type = st.radio("Select Analysis Type", ["Efficient Frontier", "Leverage Frontier"])

if analysis_type == "Efficient Frontier":
    plot_efficient_frontier()
else:
    plot_leverage_frontier()
