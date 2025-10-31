import streamlit as st
import plotly.graph_objects as go
import numpy as np
from noise import pnoise2  # pip install noise

st.set_page_config(layout="centered")
st.title("IV Chart Dashboard with Biomorphic Fractal Deformation")

# --- User input: deformation factor ---
factor = st.slider("Deformation Level (Complexity)", 0, 100, 0)  # visual 0–100
fractal_strength = 3 * (factor / 100) ** 1.5  # internal 0–3 range

# --- Constants ---
num_points = 500
max_radius_iv = 0.5   # radius of IV chart
outer_base_radius = 1.0

theta = np.linspace(0, 2 * np.pi, num_points)

# --- Start with a SQUARE in polar coordinates ---
r_square = outer_base_radius / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))

# --- Perlin noise–based deformation (biomorphic) ---
def perlin_biomorphic_radius(theta, base_r, strength):
    # smooth deformation that increases with strength
    r = np.zeros_like(theta)
    for i, t in enumerate(theta):
        # sample Perlin noise field with circular mapping
        nx, ny = np.cos(t) * (1.5 + strength), np.sin(t) * (1.5 + strength)
        r[i] = base_r[i] * (1 + 0.15 * strength * pnoise2(nx, ny, octaves=3))
    return r

r_deformed = perlin_biomorphic_radius(theta, r_square, fractal_strength)

# --- Roundening: transition from square → circular biomorphic ---
roundness = np.exp(-fractal_strength * 0.5)
r_smooth = r_deformed * (1 - roundness) + roundness * 1.0  # 1.0 ~ circle radius

# --- Clip and convert to Cartesian coordinates ---
r_smooth = np.clip(r_smooth, max_radius_iv * 1.3, None)
x = r_smooth * np.cos(theta)
y = r_smooth * np.sin(theta)

# --- Plot setup ---
fig = go.Figure()

# Outer biomorphic fractal shape
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        fill="toself",
        mode="lines",
        line=dict(color="lightblue", width=3),
        name="Outer shape",
    )
)

# --- IV Chart ---
num_sections = 6
num_vars = 9
num_levels = 5

# Concentric rings
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

# Sector dividers
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

# Variable spokes
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

# Outer boundary of IV chart
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
    width=700,
    height=700,
    xaxis=dict(scaleanchor="y", visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

