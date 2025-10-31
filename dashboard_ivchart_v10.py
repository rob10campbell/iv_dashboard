import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="centered")
st.title("IV Chart Dashboard: Square → Geometric Flower Morph")

# --- User input: deformation factor ---
factor = st.slider("Morph Level (0 = Square, 100 = Flower)", 0, 100, 0)
morph_strength = factor / 100  # normalized [0, 1]

# --- Constants ---
num_points = 500
max_radius_iv = 0.5   # radius of IV chart
outer_base_radius = 1.0

theta = np.linspace(0, 2 * np.pi, num_points)

# --- Start as square (in polar coordinates) ---
r_square = outer_base_radius / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))

# --- Target shape: geometric “flower” pattern ---
# Uses cos(4θ) to generate 4-lobed symmetry from the square's sides
r_flower = outer_base_radius * (1 + 0.1 * np.cos(4 * theta) + 0.05 * np.cos(8 * theta))

# --- Interpolate between square and flower ---
r_morph = (1 - morph_strength) * r_square + morph_strength * r_flower

# --- Convert to Cartesian coordinates ---
x = r_morph * np.cos(theta)
y = r_morph * np.sin(theta)

# --- Plot setup ---
fig = go.Figure()

# Outer morphing shape
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

# --- IV Chart (same as before) ---
num_sections = 6
num_vars = 9
num_levels = 5

# Concentric circles
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
    width=500,  # 50% smaller as you asked
    height=500,
    xaxis=dict(scaleanchor="y", visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

