import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="centered")
st.title("IV Chart Dashboard with Fractal Deformation")

# --- User input: deformation factor ---
factor = st.slider("Deformation Level (Complexity)", 0, 10, 0)

# --- Constants ---
num_points = 500
max_radius_iv = 0.5   # radius of IV chart
outer_base_radius = 1.0

theta = np.linspace(0, 2 * np.pi, num_points)
r_square = outer_base_radius / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))

# --- Fractal-like deformation function ---
def fractal_radius(theta, base_r, amplitude, levels, seed=0):
    np.random.seed(seed)
    r = np.zeros_like(theta)
    for n in range(levels):
        freq = 2 ** n
        phase = np.random.uniform(0, 2 * np.pi)
        r += (np.sin(freq * theta + phase) / (2 ** n))
    r = base_r + amplitude * r / np.max(np.abs(r))
    return r

# Deformation: number of levels increases with slider
amplitude = 0.03
levels = max(1, factor)  # 1–10 fractal layers
r_deformed = fractal_radius(theta, r_square, amplitude, levels)

# Keep shape within reasonable size
r_deformed = np.clip(r_deformed, max_radius_iv * 1.3, None)
x = r_deformed * np.cos(theta)
y = r_deformed * np.sin(theta)

# --- Plot setup ---
fig = go.Figure()

# Outer fractal shape
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

# Outer boundary
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

