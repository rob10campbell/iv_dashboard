import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="centered")
st.title("IV Chart Dashboard with Deforming Outer Shape")

# --- User input: deformation factor ---
factor = st.slider("Deformation Level", 0, 10, 0)

# --- 1. Generate "flower" outer shape ---
theta = np.linspace(0, 2 * np.pi, 300)
r = 1 + 0.2 * factor * np.sin(6 * theta)  # 6-petal deformation
x = r * np.cos(theta)
y = r * np.sin(theta)

# --- 2. Create Plotly figure ---
fig = go.Figure()

# Draw outer flower-like shape
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

# --- 3. Draw IV Chart (inner circle) ---
num_sections = 6
num_vars = 9
num_levels = 5
max_radius = 0.5

# Concentric circles (5 radial bars)
for j in range(1, num_levels + 1):
    r_ring = max_radius * (j / num_levels)
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

# Sector dividers (6 sections)
for i in range(num_sections):
    angle = i * (2 * np.pi / num_sections)
    fig.add_trace(
        go.Scatter(
            x=[0, max_radius * np.cos(angle)],
            y=[0, max_radius * np.sin(angle)],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )

# Variable spokes within each section
for i in range(num_sections):
    start_angle = i * (2 * np.pi / num_sections)
    end_angle = (i + 1) * (2 * np.pi / num_sections)
    section_width = end_angle - start_angle
    for v in range(1, num_vars + 1):
        # Evenly space 9 spokes inside each sector
        angle = start_angle + (v / (num_vars + 1)) * section_width
        fig.add_trace(
            go.Scatter(
                x=[0, max_radius * np.cos(angle)],
                y=[0, max_radius * np.sin(angle)],
                mode="lines",
                line=dict(color="lightgray", width=1),
                showlegend=False,
            )
        )

# Outer boundary of IV chart
theta_circ = np.linspace(0, 2 * np.pi, 300)
fig.add_trace(
    go.Scatter(
        x=max_radius * np.cos(theta_circ),
        y=max_radius * np.sin(theta_circ),
        mode="lines",
        line=dict(color="black", width=3),
        showlegend=False,
    )
)

# --- 4. Final layout ---
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

