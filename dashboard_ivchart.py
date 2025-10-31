import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="centered")
st.title("IV Chart Dashboard — T-Square Fractal Outer Shape")

# --- User controls ---
iterations = st.slider("Fractal iterations (T-square)", 0, 6, 0)
jitter = st.slider("Jitter (random seed variation)", 0.0, 1.0, 0.0, step=0.01)
max_radius_iv = 0.5   # inner IV circle radius (constant)
outer_base_radius = 1.0  # max half-width / bounding radius

# --- helper: generate T-square small squares ---
def generate_t_squares(iterations, base_half_size):
    """
    Returns list of squares as (cx, cy, half_size).
    T-square rule: start with center square; at each iteration, for every square
    from the previous iteration add 4 squares centered at the midpoints of the
    four edges with half the size.
    """
    squares = [(0.0, 0.0, base_half_size)]
    prev_layer = [(0.0, 0.0, base_half_size)]
    for it in range(iterations):
        new_layer = []
        for (cx, cy, h) in prev_layer:
            # midpoints of edges -> centers for new squares
            new_layer.append((cx + h, cy, h / 2.0))
            new_layer.append((cx - h, cy, h / 2.0))
            new_layer.append((cx, cy + h, h / 2.0))
            new_layer.append((cx, cy - h, h / 2.0))
        squares.extend(new_layer)
        prev_layer = new_layer
    return squares

# --- helper: ray-square intersection (returns smallest positive t or None) ---
def ray_intersect_square(theta, cx, cy, halfsize):
    """
    Ray: p = t*(cos, sin), t>0
    Square bounds: [cx-halfsize, cx+halfsize] x [cy-halfsize, cy+halfsize]
    Compute t intersections with the four lines x = cx +/- halfsize and y = cy +/- halfsize,
    keep those where the other coordinate lies within the square interval.
    Return the smallest positive t (closest intersection), or None if no intersection.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    t_candidates = []

    # vertical sides x = cx +/- halfsize
    for x_side in (cx - halfsize, cx + halfsize):
        if abs(c) > 1e-12:
            t = x_side / c
            if t > 1e-12:
                y_at = t * s
                if (y_at >= cy - halfsize - 1e-9) and (y_at <= cy + halfsize + 1e-9):
                    t_candidates.append(t)
    # horizontal sides y = cy +/- halfsize
    for y_side in (cy - halfsize, cy + halfsize):
        if abs(s) > 1e-12:
            t = y_side / s
            if t > 1e-12:
                x_at = t * c
                if (x_at >= cx - halfsize - 1e-9) and (x_at <= cx + halfsize + 1e-9):
                    t_candidates.append(t)

    if not t_candidates:
        return None
    # return the smallest positive t (closest intersection of ray with that square)
    return min(t_candidates)

# --- Build squares ---
base_half_size = outer_base_radius * 0.9  # keep a small margin inside bounding radius
squares = generate_t_squares(iterations, base_half_size)

# optional deterministic jitter to break perfect symmetry
rng = np.random.RandomState(int(jitter * 1e6))
if jitter > 0:
    squares = [(cx + rng.normal(scale=0.002*half), cy + rng.normal(scale=0.002*half), half * (1 + rng.normal(scale=0.01)))
               for (cx, cy, half) in squares]

# --- Ray-march envelope: find farthest intersection along each theta ---
num_points = 800
theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
radii = np.full_like(theta, 0.0)

for i, th in enumerate(theta):
    t_max = 0.0
    for (cx, cy, half) in squares:
        t = ray_intersect_square(th, cx, cy, half)
        if t is not None and t > t_max:
            t_max = t
    # fallback if no intersection found (shouldn't happen) -> use base radius
    if t_max <= 0:
        t_max = outer_base_radius
    radii[i] = t_max

# --- Ensure the IV circle fits comfortably inside the fractal envelope ---
min_envelope_radius = np.min(radii)
required_min = max_radius_iv * 1.35  # safety margin
if min_envelope_radius < required_min:
    # shift radii outward proportionally where needed so the min >= required_min
    shift = required_min - min_envelope_radius
    radii += shift

# small smoothing to remove tiny jagged artifacts on the final plotted line
from scipy.ndimage import gaussian_filter1d
radii_smooth = gaussian_filter1d(radii, sigma=2)

x = radii_smooth * np.cos(theta)
y = radii_smooth * np.sin(theta)

# --- build figure ---
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=x, y=y, fill="toself", mode="lines",
        line=dict(color="lightblue", width=2),
        name="T-square fractal envelope"
    )
)

# --- IV Chart (unchanged) ---
num_sections = 6
num_vars = 9
num_levels = 5

# concentric rings
for j in range(1, num_levels + 1):
    r_ring = max_radius_iv * (j / num_levels)
    theta_ring = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(
        go.Scatter(
            x=r_ring * np.cos(theta_ring),
            y=r_ring * np.sin(theta_ring),
            mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False
        )
    )

# section dividers
for i in range(num_sections):
    angle = i * (2 * np.pi / num_sections)
    fig.add_trace(
        go.Scatter(
            x=[0, max_radius_iv * np.cos(angle)],
            y=[0, max_radius_iv * np.sin(angle)],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False
        )
    )

# variable spokes
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
                showlegend=False
            )
        )

# IV circle boundary
theta_circ = np.linspace(0, 2 * np.pi, 300)
fig.add_trace(
    go.Scatter(
        x=max_radius_iv * np.cos(theta_circ),
        y=max_radius_iv * np.sin(theta_circ),
        mode="lines",
        line=dict(color="black", width=3),
        showlegend=False
    )
)

fig.update_layout(
    width=750, height=750,
    xaxis=dict(scaleanchor="y", visible=False),
    yaxis=dict(visible=False),
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="white",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

