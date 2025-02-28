import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
import cvxpy as cp
import warnings
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yfinance as yf

# Set Seaborn theme
sns.set_theme(context="talk", style="whitegrid", palette="colorblind", color_codes=True, rc={"figure.figsize": [12, 8]})

# Define 20 diversified assets across various industries
ASSETS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META",    # Technology
    "NFLX", "ADSK", "INTC", "NVDA", "TSM",      # Semiconductors & Tech
    "JPM", "GS", "BAC", "C", "AXP",            # Finance
    "UNH", "LLY", "PFE", "BMY", "MRK"          # Healthcare
]

# Streamlit configuration
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

# Streamlit Title
st.title("Portfolio Optimization")

# Sidebar for filters
st.sidebar.header("Filters")

# Filter for Asset Selection
selected_assets = st.sidebar.multiselect(
    "Select Assets",
    ASSETS,
    default=ASSETS  # Default to all assets
)

# Filter for Date Range
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2017-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Download the data
prices_df = yf.download(selected_assets, start=start_date, end=end_date)

# Ensure the data is downloaded correctly
if prices_df.empty:
    st.write("Data could not be fetched. Please check the selected assets and date range.")
else:
    # Show the last few rows of the data
    st.write(f"Downloaded data for {len(selected_assets)} assets from {start_date} to {end_date}")
    st.write(prices_df.tail())

    # Plot the asset prices
    st.subheader("Asset Prices Over Time")
    fig, ax = plt.subplots()
    prices_df["Close"].plot(ax=ax)
    ax.set(title="Asset Prices Over Time", xlabel="Date", ylabel="Price (USD)")
    sns.despine()
    st.pyplot(fig)

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

# Set Seaborn theme
sns.set_theme(context="talk", style="whitegrid", palette="colorblind", color_codes=True, rc={"figure.figsize": [12, 8]})

# Define 20 diversified assets across various industries
ASSETS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META",    # Technology
    "NFLX", "ADSK", "INTC", "NVDA", "TSM",      # Semiconductors & Tech
    "JPM", "GS", "BAC", "C", "AXP",            # Finance
    "UNH", "LLY", "PFE", "BMY", "MRK"          # Healthcare
]

# Streamlit configuration
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

# Streamlit Title
st.title("Portfolio Optimization")

# Sidebar for filters
st.sidebar.header("Filters")

# Filter for Asset Selection
selected_assets = st.sidebar.multiselect(
    "Select Assets",
    ASSETS,
    default=ASSETS  # Default to all assets
)

# Filter for Date Range
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2017-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Filter for Gamma (Risk Aversion)
gamma_value = st.sidebar.slider("Risk Aversion (Gamma)", min_value=0.001, max_value=100.0, value=0.1, step=0.001)

# Filter for Leverage Levels
max_leverage_value = st.sidebar.slider("Max Leverage", min_value=1, max_value=5, value=1, step=1)

# Download the data
prices_df = yf.download(selected_assets, start=start_date, end=end_date)
returns = prices_df["Close"].pct_change().dropna()

# Ensure the data is downloaded correctly
if prices_df.empty:
    st.write("Data could not be fetched. Please check the selected assets and date range.")
else:
    # Show the last few rows of the data
    st.write(f"Downloaded data for {len(selected_assets)} assets from {start_date} to {end_date}")
    st.write(prices_df.tail())

    # Portfolio weights
    portfolio_weights = len(selected_assets) * [1 / len(selected_assets)]
    portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index=returns.index)

    # Plot portfolio performance (first graph)
    st.subheader("1/n Portfolio Performance")
    fig1, ax1 = plt.subplots()
    qs.plots.snapshot(portfolio_returns, title="1/n portfolio's performance", grayscale=True, ax=ax1)
    st.pyplot(fig1)

    # Display portfolio metrics (second graph)
    st.subheader("Portfolio Metrics")
    fig2, ax2 = plt.subplots()
    qs.reports.metrics(portfolio_returns, benchmark="SPY", mode="basic")
    st.pyplot(fig2)

    # Optimization
    avg_returns = returns.mean().values
    cov_mat = returns.cov().values

    weights = cp.Variable(len(selected_assets))
    gamma_par = cp.Parameter(nonneg=True)
    portf_rtn_cvx = avg_returns @ weights
    portf_vol_cvx = cp.quad_form(weights, cov_mat)
    objective_function = cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx)
    problem = cp.Problem(objective_function, [cp.sum(weights) == 1, weights >= 0])

    # Efficient frontier calculation
    N_POINTS = 25
    portf_rtn_cvx_ef = []
    portf_vol_cvx_ef = []
    weights_ef = []
    gamma_range = np.logspace(-3, 3, num=N_POINTS)

    for gamma in gamma_range:
        gamma_par.value = gamma_value  # Use the selected gamma value
        problem.solve()
        portf_vol_cvx_ef.append(cp.sqrt(portf_vol_cvx).value)
        portf_rtn_cvx_ef.append(portf_rtn_cvx.value)
        weights_ef.append(weights.value)

    weights_df = pd.DataFrame(weights_ef, columns=selected_assets, index=np.round(gamma_range, 3))

    # Plot weights allocation (third graph)
    st.subheader("Weights Allocation per Risk-Aversion Level")
    fig3, ax3 = plt.subplots()
    weights_df.plot(kind="bar", stacked=True, ax=ax3)
    ax3.set(title="Weights allocation per risk-aversion level", xlabel=r"$\gamma$", ylabel="Weight")
    ax3.legend(bbox_to_anchor=(1, 1))
    sns.despine()
    st.pyplot(fig3)

    # Plot Efficient Frontier (fourth graph)
    st.subheader("Efficient Frontier")
    fig4, ax4 = plt.subplots()
    ax4.plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, "g-")
    MARKERS = ["o", "X", "d", "*"]
    for asset_index in range(len(selected_assets)):
        ax4.scatter(x=np.sqrt(cov_mat[asset_index, asset_index]),
                    y=avg_returns[asset_index],
                    marker=MARKERS[asset_index % len(MARKERS)],
                    label=selected_assets[asset_index],
                    s=150)
    ax4.set(title="Efficient Frontier", xlabel="Volatility", ylabel="Expected Returns")
    ax4.legend()
    sns.despine()
    st.pyplot(fig4)

    # Plot efficient frontier with leverage (fifth graph)
    st.subheader("Efficient Frontier for Different Max Leverage")
    max_leverage = cp.Parameter()
    prob_with_leverage = cp.Problem(objective_function, [cp.sum(weights) == 1, cp.norm(weights, 1) <= max_leverage])
    LEVERAGE_RANGE = [max_leverage_value]
    len_leverage = len(LEVERAGE_RANGE)

    portf_vol_l = np.zeros((N_POINTS, len_leverage))
    portf_rtn_l = np.zeros((N_POINTS, len_leverage))
    weights_ef = np.zeros((len_leverage, N_POINTS, len(selected_assets)))

    for lev_ind, leverage in enumerate(LEVERAGE_RANGE):
        for gamma_ind in range(N_POINTS):
            max_leverage.value = leverage
            gamma_par.value = gamma_range[gamma_ind]
            prob_with_leverage.solve()
            portf_vol_l[gamma_ind, lev_ind] = cp.sqrt(portf_vol_cvx).value
            portf_rtn_l[gamma_ind, lev_ind] = portf_rtn_cvx.value
            weights_ef[lev_ind, gamma_ind, :] = weights.value

    fig5, ax5 = plt.subplots()
    for leverage_index, leverage in enumerate(LEVERAGE_RANGE):
        ax5.plot(portf_vol_l[:, leverage_index], portf_rtn_l[:, leverage_index], label=f"{leverage}")
    ax5.set(title="Efficient Frontier for different max leverage", xlabel="Volatility", ylabel="Expected Returns")
    ax5.legend(title="Max leverage")
    sns.despine()
    st.pyplot(fig5)

    # Plot weight allocation for different leverage levels (sixth graph)
    st.subheader("Weight Allocation per Risk-Aversion Level for Different Leverage")
    fig6, ax6 = plt.subplots(len_leverage, 1, sharex=True)
    for ax_index in range(len_leverage):
        weights_df = pd.DataFrame(weights_ef[ax_index], columns=selected_assets, index=np.round(gamma_range, 3))
        weights_df.plot(kind="bar", stacked=True, ax=ax6[ax_index], legend=None)
        ax6[ax_index].set(ylabel=(f"max_leverage = {LEVERAGE_RANGE[ax_index]}\n weight"))
    ax6[len_leverage - 1].set(xlabel=r"$\gamma$")
    ax6[0].legend(bbox_to_anchor=(1, 1))
    ax6[0].set_title("Weights allocation per risk-aversion level", fontsize=16)
    sns.despine()
    st.pyplot(fig6)
