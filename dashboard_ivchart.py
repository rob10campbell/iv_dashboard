import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter1d

st.set_page_config(layout="centered")
st.title("Mandala Generator 🌸")

# --- User Controls ---
shape = st.selectbox("Base Shape", ["Circle", "Polygon", "Square"])
n_sides = st.slider("Number of sides / petals", 3, 40, 8)
flower_amp = st.slider("Flower amplitude", 0.0, 0.3, 0.1)
flower_freq = st.slider("Flower frequency", 1, 30, 8)
noise_level = st.slider("Organic noise", 0.0, 0.2, 0.05)
smoothness = st.slider("Smoothness", 0.0, 1.0, 0.3)
rotation = st.slider("Rotation angle", 0, 360, 0)

# --- Constants ---
num_points = 6000
theta = np.linspace(0, 2*np.pi, num_points)
theta_rot = theta + np.deg2rad(rotation)
outer_base_radius = 1.0

# --- Base radius ---
if shape == "Square":
    r_base = outer_base_radius / np.maximum(np.abs(np.cos(theta_rot)), np.abs(np.sin(theta_rot)))
elif shape == "Polygon":
    r_base = outer_base_radius / np.cos((np.pi/n_sides) - (theta_rot % (2*np.pi/n_sides)))
else:
    r_base = np.ones_like(theta_rot)

# --- Flower-like harmonic pattern ---
r_flower = 1 + flower_amp * np.cos(flower_freq * theta_rot)

# --- Organic noise ---
rng = np.random.default_rng(42)
noise = rng.normal(0, 1, num_points)
sigma = 2 + 80 * smoothness
noise_smooth = gaussian_filter1d(noise, sigma=sigma)
noise_smooth /= np.max(np.abs(noise_smooth))
r_noise = 1 + noise_level * noise_smooth

# --- Final radius ---
r_final = r_base * r_flower * r_noise

# --- Cartesian coords ---
x = r_final * np.cos(theta)
y = r_final * np.sin(theta)

# --- Plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=y, mode="lines", fill="toself",
    line=dict(color="royalblue", width=3),
    fillcolor="lightblue"
))

fig.update_layout(
    width=400, height=400,
    xaxis=dict(scaleanchor="y", visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

