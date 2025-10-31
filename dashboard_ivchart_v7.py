import streamlit as st
import plotly.graph_objects as go
import numpy as np
from noise import pnoise2  # pip install noise

st.set_page_config(page_title="Biomorphic Fractal Morph", layout="wide")

st.title("🌀 Biomorphic Fractal Morph Explorer")

# Slider: 0–100, mapped to 0–3 (nonlinear response)
slider_val = st.slider("Fractal Intensity", 0, 100, 50)
fractal_strength = 3 * (slider_val / 100) ** 1.5  # nonlinear mapping

n_points = 400
theta = np.linspace(0, 2 * np.pi, n_points)

# Perlin noise parameters
scale = 1.5
noise_scale = 1.0 + fractal_strength * 0.5
amp = 0.5 + fractal_strength * 0.4

# Generate smooth Perlin field along circle
r_base = 1.0
r_noise = np.array([
    r_base
    + amp * pnoise2(np.cos(t) * noise_scale, np.sin(t) * noise_scale, octaves=3)
    for t in theta
])

# Roundening (soft curvature)
roundness = np.exp(-fractal_strength * 0.5)
r = r_noise * (1 - 0.3 * roundness) + roundness

# Convert to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)

# Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=3, color="teal")))

fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    width=700,
    height=700,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="black",
    paper_bgcolor="black",
)

st.plotly_chart(fig, use_container_width=True)

