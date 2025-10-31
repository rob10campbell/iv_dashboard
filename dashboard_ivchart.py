import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter1d

st.set_page_config(layout="centered")
st.title("IV Chart Dashboard: Square → Symmetric Organic 20-gon Morph")

# --- User input ---
factor = st.slider("Morph Level (0 = Square, 100 = Organic 20-gon)", 0, 100, 0)
morph_strength = factor / 100

# --- Fixed fractal smoothness ---
smooth_strength = 23 / 100  # fixed at your preferred roughness

# --- Constants ---
num_points = 1500
max_radius_iv = 0.5
outer_base_radius = 1.0
theta = np.linspace(0, 2 * np.pi, num_points)

# --- Helper: Polar radius for regular n-gon ---
def polygon_radius(theta, n_sides):
    return outer_base_radius / np.cos((np.pi / n_sides) - (theta % (2*np.pi / n_sides)))

# --- Base shapes ---
r_square = outer_base_radius / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
r_poly = polygon_radius(theta, 20)  # use 20 sides now
r_morph = (1 - morph_strength) * r_square + morph_strength * r_poly

# --- Organic symmetric deformation ---
detail_amp = 0.05 * morph_strength
base_freq = 20  # enforce 20-fold symmetry for "petals"

# Controlled random noise (seeded)
rng = np.random.default_rng(42)
noise = rng.normal(0, 1, num_points)
sigma = 2 + 60 * smooth_strength
noise_smooth = gaussian_filter1d(noise, sigma=sigma)
noise_smooth /= np.max(np.abs(noise_smooth))

# Add symmetric harmonic modulation
# ensures fractal noise respects 20-fold rotational symmetry
harmonic = (
    np.cos(base_freq * theta)
    + 0.3 * np.sin(2 * base_freq * theta + np.pi / 5)
    + 0.15 * np.cos(4 * base_freq * theta + np.pi / 3)
)

# Mix symmetric and fractal terms
r_detail = 1 + detail_amp * (0.7 * harmonic + 0.3 * noise_smooth)

r_final = r_morph * r_detail

# --- Convert to Cartesian ---
x = r_final * np.cos(theta)
y = r_final * np.sin(theta)

# --- Plot ---
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        fill="toself",
        mode="lines",
        line=dict(color="royalblue", width=3),
        fillcolor="lightblue",
        name="Outer shape",
    )
)

# --- IV Chart (unchanged) ---
num_sections = 6
num_vars = 9
num_levels = 5

for j in range(1, num_levels + 1):
    r_ring = max_radius_iv * (j / num_levels)
    theta_ring = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=r_ring * np.cos(theta_ring),
            y=r_ring * np.sin(theta_ring),
            mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False,
        )
    )

for i in range(num_sections):
    angle = i * (2 * np.pi / num_sections)
    fig.add_trace(
        go.Scatter(
            x=[0, max_radius_iv * np.cos(angle)],
            y=[0, max_radius_iv * np.sin(angle)],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )

for i in range(num_sections):
    start_angle = i * (2 * np.pi / num_sections)
    end_angle = (i + 1) * (2 * np.pi / num_sections)
    section_width = end_angle - start_angle
    for v in range(1, num_vars + 1):
        angle = start_angle + (v / (num_vars + 1)) * section_width
        fig.add_trace(
            go.Scatter(
                x=[0, max_radius_iv * np.cos(angle)],
                y=[0, max_radius_iv * np.sin(angle)],
                mode="lines",
                line=dict(color="lightgray", width=1),
                showlegend=False,
            )
        )

theta_circ = np.linspace(0, 2 * np.pi, 300)
fig.add_trace(
    go.Scatter(
        x=max_radius_iv * np.cos(theta_circ),
        y=max_radius_iv * np.sin(theta_circ),
        mode="lines",
        line=dict(color="black", width=3),
        showlegend=False,
    )
)

# --- Layout ---
fig.update_layout(
    width=300,
    height=300,
    xaxis=dict(scaleanchor="y", visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

