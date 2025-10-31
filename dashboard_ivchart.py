import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Square-to-Flower Morph", layout="centered")

st.title("🌀 Square to Flower Morph")

# Slider goes from 0–100 but scales to 0–3 internally
flower_slider = st.slider("Flower Morph", 0, 100, 0)
flower = 3 * (flower_slider / 100)

# Shape parameters
theta = np.linspace(0, 2 * np.pi, 800)
r_base = np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))  # Start as square

# Add rounded/octagonal morph
round_factor = flower / 3  # between 0 and 1
r_shape = ((r_base ** (1 - round_factor)) * (1 - round_factor) + 1 * round_factor)

# Add flower-like lobes (sinusoidal modulation)
r_flower = r_shape * (1 + 0.1 * np.cos(4 * theta * (1 + flower)))

# Convert to Cartesian
x = r_flower * np.cos(theta)
y = r_flower * np.sin(theta)

# Make it 50% smaller
x *= 0.5
y *= 0.5

# Plot the IV circle in the center (fixed size)
circle_theta = np.linspace(0, 2 * np.pi, 200)
circle_x = 0.15 * np.cos(circle_theta)
circle_y = 0.15 * np.sin(circle_theta)

fig = go.Figure()

# Outer flower shape
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='toself',
                         line=dict(color='mediumorchid', width=3),
                         fillcolor='lavender'))

# Inner circle
fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines',
                         line=dict(color='purple', width=2)))

fig.update_layout(
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    width=500,
    height=500,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="white",
    plot_bgcolor="white"
)

st.plotly_chart(fig)

