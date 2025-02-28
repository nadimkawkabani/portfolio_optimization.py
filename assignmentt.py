import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp
import quantstats as qs

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
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')  # Legend on the far right
    st.pyplot(fig)

    # Compute returns for portfolio performance
    returns = prices_df["Close"].pct_change().dropna()
    portfolio_weights = len(selected_assets) * [1 / len(selected_assets)]
    portfolio_returns = pd.Series(
        np.dot(portfolio_weights, returns.T),
        index=returns.index
    )

    # Portfolio performance plot
    st.subheader("Portfolio Performance (1/n Portfolio)")
    fig, ax = plt.subplots()
    qs.plots.snapshot(portfolio_returns, title="1/n Portfolio's Performance", grayscale=True, ax=ax)
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')  # Legend on the far right
    st.pyplot(fig)

    # Efficient Frontier Plot
    st.subheader("Efficient Frontier")

    # Optimization setup for efficient frontier
    avg_returns = returns.mean().values
    cov_mat = returns.cov().values
    n_assets = len(selected_assets)
    
    weights = cp.Variable(n_assets)
    gamma_par = cp.Parameter(nonneg=True)
    portf_rtn_cvx = avg_returns @ weights
    portf_vol_cvx = cp.quad_form(weights, cov_mat)
    
    objective_function = cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx)
    problem = cp.Problem(objective_function, [cp.sum(weights) == 1, weights >= 0])
    
    N_POINTS = 25
    portf_rtn_cvx_ef = []
    portf_vol_cvx_ef = []
    
    gamma_range = np.logspace(-3, 3, num=N_POINTS)
    
    for gamma in gamma_range:
        gamma_par.value = gamma
        problem.solve()
        portf_vol_cvx_ef.append(np.sqrt(portf_vol_cvx).value)
        portf_rtn_cvx_ef.append(portf_rtn_cvx.value)
    
    # Plot the efficient frontier
    fig, ax = plt.subplots()
    ax.plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, "g-", label="Efficient Frontier")
    ax.set(title="Efficient Frontier", xlabel="Volatility", ylabel="Expected Returns")
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')  # Legend on the far right
    sns.despine()
    st.pyplot(fig)

    # Leverage Efficient Frontier
    st.subheader("Efficient Frontier with Leverage")

    # Leverage setup
    max_leverage = cp.Parameter()
    prob_with_leverage = cp.Problem(objective_function, [cp.sum(weights) == 1, cp.norm(weights, 1) <= max_leverage])

    LEVERAGE_RANGE = [1, 2, 5]
    len_leverage = len(LEVERAGE_RANGE)
    portf_vol_l = np.zeros((N_POINTS, len_leverage))
    portf_rtn_l = np.zeros((N_POINTS, len_leverage))
    weights_ef = np.zeros((len_leverage, N_POINTS, n_assets))

    for lev_ind, leverage in enumerate(LEVERAGE_RANGE):
        for gamma_ind in range(N_POINTS):
            max_leverage.value = leverage
            gamma_par.value = gamma_range[gamma_ind]
            prob_with_leverage.solve()
            portf_vol_l[gamma_ind, lev_ind] = np.sqrt(portf_vol_cvx).value
            portf_rtn_l[gamma_ind, lev_ind] = portf_rtn_cvx.value
            weights_ef[lev_ind, gamma_ind, :] = weights.value
    
    # Plot leverage efficient frontier
    fig, ax = plt.subplots()
    for leverage_index, leverage in enumerate(LEVERAGE_RANGE):
        plt.plot(portf_vol_l[:, leverage_index], portf_rtn_l[:, leverage_index], label=f"Leverage {leverage}")

    ax.set(title="Efficient Frontier for Different Max Leverage", xlabel="Volatility", ylabel="Expected Returns")
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')  # Legend on the far right
    sns.despine()
    st.pyplot(fig)

    # Weights Allocation per Leverage
    st.subheader("Weights Allocation per Leverage")

    fig, ax = plt.subplots(len_leverage, 1, sharex=True)

    for ax_index in range(len_leverage):
        weights_df = pd.DataFrame(weights_ef[ax_index], columns=selected_assets, index=np.round(gamma_range, 3))
        weights_df.plot(kind="bar", stacked=True, ax=ax[ax_index], legend=None)
        ax[ax_index].set(ylabel=(f"Max Leverage = {LEVERAGE_RANGE[ax_index]}\nWeight"))

    ax[len_leverage - 1].set(xlabel=r"$\gamma$")
    ax[0].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')  # Legend on the far right
    ax[0].set_title("Weights Allocation per Risk-Aversion Level", fontsize=16)

    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
