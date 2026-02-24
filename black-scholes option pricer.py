import streamlit as st
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# ---- Black–Scholes function ----
def bs_price(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put  = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return call, put

# ---- UI ----
st.title("Black–Scholes Option Pricer")

st.sidebar.header("Option Parameters")

S = st.sidebar.number_input(
    "Current Asset Price",
    min_value=0.0,
    value=100.0,
    step=1.0
)

K = st.sidebar.number_input(
    "Strike Price",
    min_value=0.0,
    value=100.0,
    step=1.0
)

T = st.sidebar.number_input(
    "Time to Maturity (Years)",
    min_value=0.01,
    value=1.0,
    step=0.01
)

sigma = st.sidebar.number_input(
    "Volatility (σ)",
    min_value=0.01,
    value=0.20,
    step=0.01
)

r = st.sidebar.number_input(
    "Risk-Free Interest Rate",
    min_value=0.0,
    value=0.05,
    step=0.005
)

call_price, put_price = bs_price(S, K, T, r, sigma)

st.subheader("Option Prices")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="
            background-color: #00ff48;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
        ">
            <div style="font-size: 18px; font-weight: 600;">CALL Value</div>
            <div style="font-size: 32px; font-weight: 800;">
                ${call_price:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="
            background-color: #F84F31;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
        ">
            <div style="font-size: 18px; font-weight: 600;">PUT Value</div>
            <div style="font-size: 32px; font-weight: 800;">
                ${put_price:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
# =========================
# Heatmap Parameters
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Heatmap Parameters")

spot_min = st.sidebar.number_input("Min Spot Price", value=80.0, step=1.0)
spot_max = st.sidebar.number_input("Max Spot Price", value=120.0, step=1.0)

vol_min = st.sidebar.slider(
    "Min Volatility",
    min_value=0.01,
    max_value=1.00,
    value=0.10,
    step=0.01
)

vol_max = st.sidebar.slider(
    "Max Volatility",
    min_value=0.01,
    max_value=1.00,
    value=0.60,
    step=0.01
)

grid_size = st.sidebar.slider(
    "Heatmap Resolution",
    min_value=5,
    max_value=30,
    value=10
)

# =========================
# Create ranges
# =========================
spot_range = np.linspace(spot_min, spot_max, grid_size)
vol_range  = np.linspace(vol_min, vol_max, grid_size)

# =========================
# Build matrices
# =========================
call_matrix = np.zeros((grid_size, grid_size))
put_matrix  = np.zeros((grid_size, grid_size))

for i, vol in enumerate(vol_range):
    for j, spot in enumerate(spot_range):
        call_matrix[i, j], put_matrix[i, j] = bs_price(
            spot, K, T, r, vol
        )

# =========================
# Heatmap helper function
# =========================
def plot_heatmap(ax, data, title, cmap):
    im = ax.imshow(
        data,
        cmap=cmap,
        origin="lower",
        aspect="auto"
    )

    ax.set_title(title)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")

    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticklabels([f"{s:.1f}" for s in spot_range], rotation=45)
    ax.set_yticklabels([f"{v:.2f}" for v in vol_range])

    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(
                j, i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

    return im

# =========================
# Plot heatmaps
# =========================
st.subheader("Option Price Sensitivity Heatmaps")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

im1 = plot_heatmap(ax1, call_matrix, "Call Price Heatmap", "RdYlGn")
fig.colorbar(im1, ax=ax1, label="Call Price")

im2 = plot_heatmap(ax2, put_matrix, "Put Price Heatmap", "RdYlGn")
fig.colorbar(im2, ax=ax2, label="Put Price")

st.pyplot(fig)
page_title="Lum-Ashton | Black-Scholes Option Pricer",
page_icon="📈",
layout="wide"

st.write("done by Ashton Lum https://www.linkedin.com/in/lum-ashton/")
