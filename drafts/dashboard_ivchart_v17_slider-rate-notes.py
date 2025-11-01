import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter1d
import io

st.set_page_config(layout="wide")
st.title("Context-aware Tasting Diagram")

# --- User input for context ---
factor = st.slider("Amount of context (0 to 100)", 0, 100, 0)
morph_strength = factor / 100
smooth_strength = 23 / 100  # fixed fractal roughness

# --- Constants ---
num_points = 8000
max_radius_iv = 0.8
outer_base_radius = 1.0
theta = np.linspace(0, 2 * np.pi, num_points)
colors = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"
]
num_sections = 6
num_vars = 9
num_levels = 6

# --- Helper: Polygon radius ---
def polygon_radius(theta, n_sides):
    return outer_base_radius / np.cos((np.pi / n_sides) - (theta % (2*np.pi / n_sides)))

# --- Base shape (smoothed morphing polygon) ---
r_square = outer_base_radius / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
r_poly = polygon_radius(theta, 20)
scale_factor = 1 + 0.25 * morph_strength
r_poly *= scale_factor
r_morph = (1 - morph_strength) * r_square + morph_strength * r_poly

# --- Add harmonic detail ---
rng = np.random.default_rng(42)
noise = rng.normal(0, 1, num_points)
sigma = 2 + 80 * smooth_strength
noise_smooth = gaussian_filter1d(noise, sigma=sigma)
noise_smooth /= np.max(np.abs(noise_smooth))
harmonic = np.cos(20 * theta) + 0.3 * np.sin(40 * theta + np.pi / 5)
r_detail = 1 + 0.05 * morph_strength * (0.7 * harmonic + 0.3 * noise_smooth)
r_final = gaussian_filter1d(r_morph * r_detail, sigma=3)
x = r_final * np.cos(theta)
y = r_final * np.sin(theta)

# --- Build plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, fill="toself", mode="lines", line=dict(color="black", width=2), fillcolor="white"))

# --- Add radial grid ---
for j in range(1, num_levels + 1):
    r_ring = max_radius_iv * (j / num_levels)
    theta_ring = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=r_ring*np.cos(theta_ring), y=r_ring*np.sin(theta_ring),
                             mode="lines", line=dict(color="black", width=0.8, dash="dot"), showlegend=False))

# --- Section lines ---
for i in range(num_sections):
    angle = i * (2 * np.pi / num_sections)
    idx = (np.abs(theta - angle)).argmin()
    r_edge = r_final[idx]
    fig.add_trace(go.Scatter(x=[0, r_edge*np.cos(angle)], y=[0, r_edge*np.sin(angle)],
                             mode="lines", line=dict(color="black", width=2), showlegend=False))

# --- IV spokes ---
angles = []
for i in range(num_sections):
    start = i * (2 * np.pi / num_sections)
    end = (i + 1) * (2 * np.pi / num_sections)
    section_width = end - start
    for v in range(1, num_vars + 1):
        angle = start + (v / (num_vars + 1)) * section_width
        angles.append(angle)
        color = colors[(v - 1) % len(colors)]
        fig.add_trace(go.Scatter(x=[0, max_radius_iv*np.cos(angle)], y=[0, max_radius_iv*np.sin(angle)],
                                 mode="lines", line=dict(color=color, width=1), showlegend=False))

# --- User selections (9 IVs × 6 levels) ---
st.sidebar.header("Select Tasting Levels")
user_levels = []
for i in range(num_vars):
    user_levels.append(st.sidebar.slider(f"Variable {i+1}", 0, num_levels, 3))

# --- Draw user polygon ---
r_points = [max_radius_iv * (lvl / num_levels) for lvl in user_levels]
x_poly = [r_points[i] * np.cos(angles[i]) for i in range(num_vars)]
y_poly = [r_points[i] * np.sin(angles[i]) for i in range(num_vars)]

# close polygon
x_poly.append(x_poly[0])
y_poly.append(y_poly[0])

fig.add_trace(go.Scatter(
    x=x_poly,
    y=y_poly,
    mode="lines+markers",
    fill="toself",
    line=dict(color="black", width=3),
    fillcolor="rgba(0,0,0,0.2)",
    marker=dict(size=6, color="black"),
    name="User Selection"
))

# --- Center cover circle ---
r_center = 0.2
theta_center = np.linspace(0, 2*np.pi, 200)
x_center = r_center * np.cos(theta_center)
y_center = r_center * np.sin(theta_center)
fig.add_trace(go.Scatter(x=x_center, y=y_center, fill="toself", mode="lines",
                         line=dict(color="black", width=2), fillcolor="white", showlegend=False))

fig.update_layout(
    width=400,
    height=400,
    xaxis=dict(scaleanchor="y", visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

# --- Download Button ---
img_bytes = fig.to_image(format="png", width=1000, height=1000, scale=2)
st.download_button("📥 Download current image", img_bytes, "tasting_diagram.png", "image/png")

