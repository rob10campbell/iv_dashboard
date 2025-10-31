import streamlit as st
import plotly.graph_objects as go
import numpy as np
from noise import pnoise2  # pip install noise

st.set_page_config(page_title="Biomorphic Fractal Morph", layout="wide")

st.title("🧬 Biomorphic Fractal Morph Explorer")

# Slider: 0–100 visible, mapped to 0–3 effective (nonlinear scaling)
slider_val = st.slider("Fractal Intensity", 0, 100, 0)
fractal_strength = 3 * (slider_val / 100) ** 1.5

n_points = 800
theta = np.linspace(0, 2 * np.pi, n_points)

# --- Start from a square shape ---
# parametric square approximation (using L∞ norm)
square_radius = np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))

# --- Add Perlin noise detail ---
scale = 1.5
noise_scale = 1.0 + fractal_strength * 0.5
amp = 0.4 * fractal_strength

r_noise = np.array([
    square_radius[i]
    + amp * pnoise2(np.cos(t) * noise_scale, np.sin(t) * noise_scale, octaves=3)
    for i, t in enumerate(theta)
])

# --- Roundening: interpolate square → circle ---
roundness = np.exp(-fractal_strength * 0.5)
r = r_noise * (1 - roundness) + roundness * 1.0  # 1.0 = circle radius

# --- Convert to Cartesian coordinates (scaled down) ---
scale_factor = 0.5  # 50% smaller
x = scale_factor * r * np.cos(theta)
y = scale_factor * r * np.sin(theta)

# --- Create plotly figure ---
fig = go.Figure()

# Add main fractal outline
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=3, color="teal")))

# Add “IV wheel” (center cross)
cross_len = 0.6 * scale_factor
fig.add_trace(go.Scatter(x=[-cross_len, cross_len], y=[0, 0],
                         mode="lines", line=dict(width=1.5, color="white", dash="dot")))
fig.add_trace(go.Scatter(x=[0, 0], y=[-cross_len, cross_len],
                         mode="lines", line=dict(width=1.5, color="white", dash="dot")))

# --- Layout ---
fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    width=400,  # 50% smaller visual size
    height=400,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="black",
    paper_bgcolor="black",
)

st.plotly_chart(fig, use_container_width=False)

