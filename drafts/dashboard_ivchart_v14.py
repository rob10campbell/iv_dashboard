import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter1d

st.set_page_config(layout="centered")
st.title("IV Chart Dashboard: Square → Smooth Organic Octagon Morph")

# --- User input ---
factor = st.slider("Morph Level (0 = Square, 100 = Organic Octagon)", 0, 100, 0)
morph_strength = factor / 100  # normalized [0, 1]

# --- Constants ---
num_points = 1500
max_radius_iv = 0.5
outer_base_radius = 1.0
theta = np.linspace(0, 2 * np.pi, num_points)

# --- Helper: Polar radius for regular n-gon ---
def polygon_radius(theta, n_sides):
    return outer_base_radius / np.cos((np.pi / n_sides) - (theta % (2*np.pi / n_sides)))

# --- Base (square) and target (octagon) shapes ---
r_square = outer_base_radius / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
r_octagon = polygon_radius(theta, 8)

# --- Interpolate between square and octagon ---
r_morph = (1 - morph_strength) * r_square + morph_strength * r_octagon

# --- Add smooth organic deformation ---
# Gentle, evolving organic ripples
base_freq = 3 + 3 * morph_strength
detail_amp = 0.04 * morph_strength

# Create low-frequency cosine wave
smooth_wave = np.cos(base_freq * theta) + 0.4 * np.sin(1.7 * base_freq * theta + np.pi / 4)

# Add gentle random noise (seeded for consistency)
rng = np.random.default_rng(42)
noise = rng.normal(0, 1, num_points)
noise = gaussian_filter1d(noise, sigma=30)  # smooth the noise

# Combine to make organic field
r_detail = 1 + detail_amp * (0.6 * smooth_wave + 0.4 * noise / np.max(np.abs(noise)))

# Apply detail
r_final = r_morph * r_detail

# --- Final smoothing of the contour ---
r_final = gaussian_filter1d(r_final, sigma=8)

# --- Convert to Cartesian coordinates ---
x = r_final * np.cos(theta)
y = r_final * np.sin(theta)

# --- Plot setup ---
fig = go.Figure()

# Outer morphing shape
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
    width=500,
    height=500,
    xaxis=dict(scaleanchor="y", visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

