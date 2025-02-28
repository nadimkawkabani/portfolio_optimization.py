import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
import cvxpy as cp
import warnings

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

n_assets = len(ASSETS)

# Streamlit configuration
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

# Streamlit Title
st.title("Portfolio Optimization")

# Download the data
prices_df = yf.download(ASSETS, start="2017-01-01", end="2023-12-31")
returns = prices_df["Close"].pct_change().dropna()

# Portfolio weights
portfolio_weights = n_assets * [1 / n_assets]
portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index=returns.index)

# Plot portfolio performance
qs.plots.snapshot(portfolio_returns, title="1/n portfolio's performance", grayscale=True)

# Display portfolio metrics
qs.reports.metrics(portfolio_returns, benchmark="SPY", mode="basic")

# Optimization
avg_returns = returns.mean().values
cov_mat = returns.cov().values

weights = cp.Variable(n_assets)
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
    gamma_par.value = gamma
    problem.solve()
    portf_vol_cvx_ef.append(cp.sqrt(portf_vol_cvx).value)
    portf_rtn_cvx_ef.append(portf_rtn_cvx.value)
    weights_ef.append(weights.value)

weights_df = pd.DataFrame(weights_ef, columns=ASSETS, index=np.round(gamma_range, 3))

# Plot weights allocation
fig, ax = plt.subplots()
weights_df.plot(kind="bar", stacked=True, ax=ax)
ax.set(title="Weights allocation per risk-aversion level", xlabel=r"$\gamma$", ylabel="Weight")
ax.legend(bbox_to_anchor=(1, 1))
sns.despine()
st.pyplot(fig)

# Plot Efficient Frontier
fig, ax = plt.subplots()
ax.plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, "g-")
MARKERS = ["o", "X", "d", "*"]
for asset_index in range(n_assets):
    ax.scatter(x=np.sqrt(cov_mat[asset_index, asset_index]),
               y=avg_returns[asset_index],
               marker=MARKERS[asset_index % len(MARKERS)],
               label=ASSETS[asset_index],
               s=150)
ax.set(title="Efficient Frontier", xlabel="Volatility", ylabel="Expected Returns")
ax.legend()
sns.despine()
st.pyplot(fig)

# Plot efficient frontier with leverage
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

fig, ax = plt.subplots()
for leverage_index, leverage in enumerate(LEVERAGE_RANGE):
    ax.plot(portf_vol_l[:, leverage_index], portf_rtn_l[:, leverage_index], label=f"{leverage}")
ax.set(title="Efficient Frontier for different max leverage", xlabel="Volatility", ylabel="Expected Returns")
ax.legend(title="Max leverage")
sns.despine()
st.pyplot(fig)

# Plot weight allocation for different leverage levels
fig, ax = plt.subplots(len_leverage, 1, sharex=True)
for ax_index in range(len_leverage):
    weights_df = pd.DataFrame(weights_ef[ax_index], columns=ASSETS, index=np.round(gamma_range, 3))
    weights_df.plot(kind="bar", stacked=True, ax=ax[ax_index], legend=None)
    ax[ax_index].set(ylabel=(f"max_leverage = {LEVERAGE_RANGE[ax_index]}\n weight"))
ax[len_leverage - 1].set(xlabel=r"$\gamma$")
ax[0].legend(bbox_to_anchor=(1, 1))
ax[0].set_title("Weights allocation per risk-aversion level", fontsize=16)
sns.despine()
st.pyplot(fig)
