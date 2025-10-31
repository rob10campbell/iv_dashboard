import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter1d

st.set_page_config(layout="centered")
st.title("IV Chart Dashboard — Smoothed Fractal Outer Shape")

# --- User controls ---
factor = st.slider("Deformation Level", 0.0, 3.0, 0.0, step=0.15)  # 0 to 3, 20 steps
max_radius_iv = 0.5   # inner IV circle radius (constant)
outer_base_radius = 1.0  # max bounding radius

# --- Generate T-square fractal squares (integer levels only) ---
def generate_t_squares(iterations, base_half_size):
    squares = [(0.0, 0.0, base_half_size)]
    prev_layer = [(0.0, 0.0, base_half_size)]
    for it in range(iterations):
        new_layer = []
        for (cx, cy, h) in prev_layer:
            new_layer.append((cx + h, cy, h / 2.0))
            new_layer.append((cx - h, cy, h / 2.0))
            new_layer.append((cx, cy + h, h / 2.0))
            new_layer.append((cx, cy - h, h / 2.0))
        squares.extend(new_layer)
        prev_layer = new_layer
    return squares

# --- Ray-square intersection helper ---
def ray_intersect_square(theta, cx, cy, halfsize):
    c = np.cos(theta)
    s = np.sin(theta)
    t_candidates = []

    for x_side in (cx - halfsize, cx + halfsize):
        if abs(c) > 1e-12:
            t = x_side / c
            if t > 1e-12:
                y_at = t * s
                if cy - halfsize <= y_at <= cy + halfsize:
                    t_candidates.append(t)
    for y_side in (cy - halfsize, cy + halfsize):
        if abs(s) > 1e-12:
            t = y_side / s
            if t > 1e-12:
                x_at = t * c
                if cx - halfsize <= x_at <= cx + halfsize:
                    t_candidates.append(t)
    return min(t_candidates) if t_candidates else None

# --- Fractal envelope builder ---
def fractal_envelope(iteration_level):
    base_half_size = outer_base_radius * 0.9
    squares = generate_t_squares(iteration_level, base_half_size)
    num_points = 800
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = np.zeros_like(theta)
    for i, th in enumerate(theta):
        t_max = 0.0
        for (cx, cy, half) in squares:
            t = ray_intersect_square(th, cx, cy, half)
            if t is not None and t > t_max:
                t_max = t
        radii[i] = t_max or outer_base_radius
    return theta, radii

# --- Generate integer fractal levels and interpolate ---
low_iter = int(np.floor(factor))
high_iter = min(low_iter + 1, 3)
frac = factor - low_iter

theta, r_low = fractal_envelope(low_iter)
_, r_high = fractal_envelope(high_iter)

# interpolate between discrete fractal levels
r_fractal = (1 - frac) * r_low + frac * r_high

# --- Morph fractal toward smoother, lobed shape ---
# Blend with a lobed circle (sinusoidal perturbation)
lobes = 6
lobed_radius = outer_base_radius * (1 + 0.05 * np.sin(lobes * theta))
blend_strength = np.clip(factor / 3.0, 0, 1)  # more smoothing as factor increases
r_morphed = (1 - blend_strength) * r_fractal + blend_strength * lobed_radius

# --- Smooth the final outline progressively ---
smooth_sigma = 2 + 8 * blend_strength
r_smooth = gaussian_filter1d(r_morphed, sigma=smooth_sigma)

# --- Ensure the IV circle fits inside ---
min_envelope_radius = np.min(r_smooth)
if min_envelope_radius < max_radius_iv * 1.3:
    r_smooth += (max_radius_iv * 1.3 - min_envelope_radius)

x = r_smooth * np.cos(theta)
y = r_smooth * np.sin(theta)

# --- Plot setup ---
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=y, fill="toself", mode="lines",
    line=dict(color="lightblue", width=3),
    name="Fractal outline"
))

# --- IV Chart ---
num_sections = 6
num_vars = 9
num_levels = 5

for j in range(1, num_levels + 1):
    r_ring = max_radius_iv * (j / num_levels)
    theta_ring = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(go.Scatter(
        x=r_ring * np.cos(theta_ring),
        y=r_ring * np.sin(theta_ring),
        mode="lines",
        line=dict(color="gray", width=1, dash="dot"),
        showlegend=False
    ))

for i in range(num_sections):
    angle = i * (2 * np.pi / num_sections)
    fig.add_trace(go.Scatter(
        x=[0, max_radius_iv * np.cos(angle)],
        y=[0, max_radius_iv * np.sin(angle)],
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False
    ))

for i in range(num_sections):
    start_angle = i * (2 * np.pi / num_sections)
    end_angle = (i + 1) * (2 * np.pi / num_sections)
    section_width = end_angle - start_angle
    for v in range(1, num_vars + 1):
        angle = start_angle + (v / (num_vars + 1)) * section_width
        fig.add_trace(go.Scatter(
            x=[0, max_radius_iv * np.cos(angle)],
            y=[0, max_radius_iv * np.sin(angle)],
            mode="lines",
            line=dict(color="lightgray", width=1),
            showlegend=False
        ))

theta_circ = np.linspace(0, 2 * np.pi, 300)
fig.add_trace(go.Scatter(
    x=max_radius_iv * np.cos(theta_circ),
    y=max_radius_iv * np.sin(theta_circ),
    mode="lines",
    line=dict(color="black", width=3),
    showlegend=False
))

fig.update_layout(
    width=750, height=750,
    xaxis=dict(scaleanchor="y", visible=False),
    yaxis=dict(visible=False),
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="white",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

