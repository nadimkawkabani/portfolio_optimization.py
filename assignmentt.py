import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

# Set Seaborn theme for the plots
sns.set_theme(context="talk", style="whitegrid", palette="colorblind", color_codes=True, rc={"figure.figsize": [12, 8]})

# Example DataFrame (you can replace this with your own data)
data = {
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Value': [23, 45, 56, 78, 33]
}
df = pd.DataFrame(data)

# Streamlit page configuration
st.set_page_config(page_title="Data Visualization", layout="wide")

# Streamlit title
st.title("Streamlit Data Visualization Example")

# Display the DataFrame
st.write("Here is an example DataFrame:")
st.dataframe(df)

# Plot using Seaborn
fig, ax = plt.subplots()
sns.barplot(x="Category", y="Value", data=df, ax=ax)
ax.set_title("Bar Plot Example")
sns.despine()

# Display the plot in Streamlit
st.pyplot(fig)

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yfinance as yf
import quantstats as qs
import cvxpy as cp

# Set up Streamlit page configuration
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

# Define diversified assets across various industries
ASSETS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META",    # Technology
    "NFLX", "ADSK", "INTC", "NVDA", "TSM",      # Semiconductors & Tech
    "JPM", "GS", "BAC", "C", "AXP",            # Finance
    "UNH", "LLY", "PFE", "BMY", "MRK"          # Healthcare
]
n_assets = len(ASSETS)

# Download data using yfinance
@st.cache
def get_data(assets, start, end):
    # Download the data for the given assets, start and end dates
    prices_df = yf.download(assets, start=start, end=end)
    
    # Extract the adjusted close prices from the multi-level columns
    adj_close = prices_df['Adj Close']
    
    return adj_close

# Efficient frontier plot function
def plot_efficient_frontier():
    # Get the data for the selected assets
    prices_df = get_data(ASSETS, "2017-01-01", "2023-12-31")
    
    # Access the adjusted close prices and calculate daily returns
    returns = prices_df.pct_change().dropna()
    
    # Portfolio optimization using cvxpy
    avg_returns = returns.mean().values
    cov_mat = returns.cov().values
    
    weights = cp.Variable(n_assets)
    gamma_par = cp.Parameter(nonneg=True)
    portf_rtn_cvx = avg_returns @ weights
    portf_vol_cvx = cp.quad_form(weights, cov_mat)
    objective_function = cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx)
    
    problem = cp.Problem(objective_function, [cp.sum(weights) == 1, weights >= 0])
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

    # Plot Efficient Frontier
    fig, ax = plt.subplots()
    ax.plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, "g-")

    MARKERS = ["o", "X", "d", "*"]
    for asset_index in range(n_assets):
        ax.scatter(
            x=np.sqrt(cov_mat[asset_index, asset_index]),
            y=avg_returns[asset_index],
            marker=MARKERS[asset_index % len(MARKERS)],
            label=ASSETS[asset_index],
            s=150
        )
    ax.set(title="Efficient Frontier", xlabel="Volatility", ylabel="Expected Returns")
    ax.legend()

    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)

# Efficient frontier for leveraged portfolios
def plot_leverage_frontier():
    # Get the data for the selected assets
    prices_df = get_data(ASSETS, "2017-01-01", "2023-12-31")
    
    # Access the adjusted close prices and calculate daily returns
    returns = prices_df.pct_change().dropna()
    
    # Portfolio optimization using cvxpy
    avg_returns = returns.mean().values
    cov_mat = returns.cov().values
    
    weights = cp.Variable(n_assets)
    gamma_par = cp.Parameter(nonneg=True)
    portf_rtn_cvx = avg_returns @ weights
    portf_vol_cvx = cp.quad_form(weights, cov_mat)
    objective_function = cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx)
    
    problem = cp.Problem(objective_function, [cp.sum(weights) == 1, weights >= 0])
    LEVERAGE_RANGE = [1, 2, 5]
    N_POINTS = 25
    portf_vol_l = np.zeros((N_POINTS, len(LEVERAGE_RANGE)))
    portf_rtn_l = np.zeros((N_POINTS, len(LEVERAGE_RANGE)))
    weights_ef = np.zeros((len(LEVERAGE_RANGE), N_POINTS, n_assets))

    for lev_ind, leverage in enumerate(LEVERAGE_RANGE):
        for gamma_ind in range(N_POINTS):
            max_leverage = cp.Parameter(value=leverage)
            prob_with_leverage = cp.Problem(
                objective_function,
                [cp.sum(weights) == 1, cp.norm(weights, 1) <= max_leverage]
            )
            gamma_par.value = np.logspace(-3, 3, num=N_POINTS)[gamma_ind]
            prob_with_leverage.solve()
            portf_vol_l[gamma_ind, lev_ind] = np.sqrt(portf_vol_cvx).value
            portf_rtn_l[gamma_ind, lev_ind] = portf_rtn_cvx.value
            weights_ef[lev_ind, gamma_ind, :] = weights.value
    
    # Plot leverage efficient frontier
    fig, ax = plt.subplots()
    for leverage_index, leverage in enumerate(LEVERAGE_RANGE):
        ax.plot(
            portf_vol_l[:, leverage_index], portf_rtn_l[:, leverage_index], label=f"{leverage}"
        )

    ax.set(title="Efficient Frontier for Different Max Leverage", xlabel="Volatility", ylabel="Expected Returns")
    ax.legend(title="Max Leverage")

    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit layout and user interaction
st.title("Portfolio Optimization and Efficient Frontier")

# Display options for the user to select the type of analysis
option = st.selectbox("Choose Analysis Type", ["Efficient Frontier", "Leverage Frontier"])

if option == "Efficient Frontier":
    plot_efficient_frontier()
else:
    plot_leverage_frontier()
