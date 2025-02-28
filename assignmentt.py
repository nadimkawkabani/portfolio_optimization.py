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

