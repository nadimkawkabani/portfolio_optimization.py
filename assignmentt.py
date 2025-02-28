import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp

# Set up the theme and figure size
sns.set_theme(context="talk", style="whitegrid", palette="colorblind", color_codes=True, rc={"figure.figsize": [12, 8]})

# Define the list of assets
ASSETS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META",    # Technology
    "NFLX", "ADSK", "INTC", "NVDA", "TSM",      # Semiconductors & Tech
    "JPM", "GS", "BAC", "C", "AXP",            # Finance
    "UNH", "LLY", "PFE", "BMY", "MRK"          # Healthcare
]
n_assets = len(ASSETS)

# Streamlit app title
st.title("Portfolio Optimization")

# Streamlit Sidebar for asset selection
selected_assets = st.sidebar.multiselect('Select Assets', ASSETS, default=ASSETS)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2017-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Download the asset price data from Yahoo Finance
prices_df = yf.download(selected_assets, start=start_date, end=end_date)

# Show the selected asset prices
st.subheader("Asset Prices")
fig, ax = plt.subplots()
prices_df["Close"].plot(ax=ax, title="Selected Assets Prices")
ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')  # Place legend on the far right
st.pyplot(fig)

# Calculate the returns of the selected assets
returns = prices_df["Close"].pct_change().dropna()

# Portfolio weights (equally distributed for the 1/n portfolio)
portfolio_weights = np.array([1 / len(selected_assets)] * len(selected_assets))

# Compute portfolio returns
portfolio_returns = pd.Series(np.dot(returns, portfolio_weights), index=returns.index)

# Display Portfolio Performance for the 1/n Portfolio
st.subheader("Portfolio Performance (1/n Portfolio)")
fig, ax = plt.subplots()
qs.plots.snapshot(portfolio_returns, title="1/n Portfolio's Performance", grayscale=True, ax=ax)
ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')  # Place legend on the far right
st.pyplot(fig)

# Efficient Frontier Calculation
avg_returns = returns.mean().values
cov_mat = returns.cov().values
weights = cp.Variable(n_assets)
gamma_par = cp.Parameter(nonneg=True)

# Use cvxpy's matmul for matrix multiplication
portf_rtn_cvx = cp.matmul(avg_returns, weights)
portf_vol_cvx = cp.quad_form(weights, cov_mat)

objective_function = cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx)

problem = cp.Problem(objective_function, [cp.sum(weights) == 1, weights >= 0])

# Generate Efficient Frontier for varying risk aversion
N_POINTS = 25
portf_rtn_cvx_ef = []
portf_vol_cvx_ef = []
gamma_range = np.logspace(-3, 3, num=N_POINTS)

for gamma in gamma_range:
    gamma_par.value = gamma
    problem.solve()
    portf_vol_cvx_ef.append(cp.sqrt(portf_vol_cvx).value)
    portf_rtn_cvx_ef.append(portf_rtn_cvx.value)

# Plot the Efficient Frontier
st.subheader("Efficient Frontier")
fig, ax = plt.subplots()
ax.plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, "g-")
ax.set(title="Efficient Frontier", xlabel="Volatility", ylabel="Expected Returns")

# Display asset points on the Efficient Frontier
MARKERS = ["o", "X", "d", "*"]
for asset_index in range(n_assets):
    ax.scatter(x=np.sqrt(cov_mat[asset_index, asset_index]),
               y=avg_returns[asset_index],
               marker=MARKERS[asset_index % len(MARKERS)],  # Use modulo to avoid IndexError
               label=selected_assets[asset_index],
               s=150)

ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')  # Place legend on the far right
sns.despine()
plt.tight_layout()
st.pyplot(fig)

# Leverage Constraints and Efficient Frontier Calculation for different max leverage
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
        portf_vol_l[gamma_ind, lev_ind] = cp.sqrt(portf_vol_cvx).value
        portf_rtn_l[gamma_ind, lev_ind] = portf_rtn_cvx.value
        weights_ef[lev_ind, gamma_ind, :] = weights.value

# Plot the Efficient Frontier for different max leverage
st.subheader("Efficient Frontier with Leverage")
fig, ax = plt.subplots()

for leverage_index, leverage in enumerate(LEVERAGE_RANGE):
    ax.plot(portf_vol_l[:, leverage_index], portf_rtn_l[:, leverage_index], label=f"Max Leverage = {leverage}")

ax.set(title="Efficient Frontier for Different Max Leverage", xlabel="Volatility", ylabel="Expected Returns")
ax.legend(title="Max Leverage", bbox_to_anchor=(1.05, 0.5), loc='center left')  # Place legend on the far right
sns.despine()
plt.tight_layout()
st.pyplot(fig)

# Plot the weights for different levels of risk-aversion (gamma) and leverage
fig, ax = plt.subplots(len_leverage, 1, sharex=True)

for ax_index in range(len_leverage):
    weights_df = pd.DataFrame(weights_ef[ax_index], columns=selected_assets, index=np.round(gamma_range, 3))
    weights_df.plot(kind="bar", stacked=True, ax=ax[ax_index], legend=None)
    ax[ax_index].set(ylabel=f"Max Leverage = {LEVERAGE_RANGE[ax_index]}\nWeight")

ax[len_leverage - 1].set(xlabel=r"$\gamma$")
ax[0].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')  # Place legend on the far right
ax[0].set_title("Weights Allocation per Risk-Aversion Level", fontsize=16)

sns.despine()
plt.tight_layout()
st.pyplot(fig)
