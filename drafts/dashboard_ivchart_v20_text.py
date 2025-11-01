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
outer_base_radius = 1.2 #1.0
theta = np.linspace(0, 2 * np.pi, num_points)
bean_colors = [ "white", "#a8b448", "#dbc649", "#c8a472", "#b8844e", "#9f6e48", "#652d17" ]  
bean_labels = [ "Uniform", "Color", "Smooth", "Shiny"]
colors = [
    "#c74a34", "#ab253f", "#a67b65", "#e45a38", 
    "#d81b6a", "#d82327", "#ebb327", "#227932", "#1ba3b4",
]
notes = [
    "Roasted", "Spices", "Nutty", "Sweet", 
    "Floral", "Fruity", "Sour", "Vegetal", "Other"
]
num_sections = 6
num_vars = 9
num_levels = 7

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
fig.add_trace(go.Scatter(x=x, y=y, fill="toself", mode="lines",
                         line=dict(color="black", width=2), fillcolor="white"))

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


# --- Adjusted section widths ---
base_angle = 2 * np.pi / num_sections
section_widths = [base_angle] * num_sections
section_widths[4] = base_angle * 0.5   # Section 5 smaller
section_widths[5] = base_angle * 1.5   # Section 6 compensates

# --- IV spokes + blank markers (section 1 customized) + labels ---
angles = []
current_angle = 0  # track running angle

for i in range(num_sections):
    #start = i * (2 * np.pi / num_sections)
    #end = (i + 1) * (2 * np.pi / num_sections)
    start = current_angle
    end = start + section_widths[i]
    current_angle = end  # increment for next loop
    section_width = end - start
    print("section", i)
    print(current_angle)
    print(section_width)

    if i == 4:
        # --- Section 5: Mask instead of diagram ---
        mask_theta = np.linspace(start, end, 100)
        mask_x = np.concatenate([[0], np.cos(mask_theta), [0]]) * outer_base_radius
        mask_y = np.concatenate([[0], np.sin(mask_theta), [0]]) * outer_base_radius

        fig.add_trace(go.Scatter(
            x=mask_x,
            y=mask_y,
            fill="toself",
            mode="lines",
            line=dict(color="white", width=0),
            fillcolor="white",
            showlegend=False
        ))
        continue  # Skip any IV drawing here

    elif i == 0:
        # --- Section 1 special case: 4 spokes, black lines ---
        num_spokes = 4
        for v in range(1, num_spokes + 1):
            angle = start + (v / (num_spokes + 1)) * section_width
            angles.append(angle)

            # spoke line (black)
            fig.add_trace(go.Scatter(
                x=[0, max_radius_iv * np.cos(angle)],
                y=[0, max_radius_iv * np.sin(angle)],
                mode="lines",
                line=dict(color="black", width=1.5),
                showlegend=False
            ))

            # blank markers for all levels
            for lvl in range(1, num_levels + 1):
                r_lvl = max_radius_iv * (lvl / num_levels)
                x_m = r_lvl * np.cos(angle)
                y_m = r_lvl * np.sin(angle)

                if v == 3: # numbers are reverse of the real order
                    # first spoke: colored level markers
                    color = bean_colors[(lvl - 1) % len(bean_colors)]
                    bean_width = 2
                else:
                    # other section-1 spokes: black markers
                    color = "black"
                    bean_width = 0.8

                fig.add_trace(go.Scatter(
                    x=[x_m], y=[y_m],
                    mode="markers",
                    marker=dict(size=6, color="white", line=dict(color=color, width=bean_width)),
                    showlegend=False
                ))

            # --- Add label using annotation instead of textangle in Scatter ---
            label_r = max_radius_iv * 1.13
            label_x = label_r * np.cos(angle)
            label_y = label_r * np.sin(angle)
            #lbl = bean_labels[v - 1] if (v - 1) < len(bean_labels) else f"S1-{v}"
            lbl = bean_labels[len(bean_labels) - v] if (len(bean_labels) - v) < len(bean_labels) else f"S1-{v}"

            # compute rotation
            text_angle = np.degrees(angle)
            if np.cos(angle) < 0:
                text_angle += 180

            fig.add_annotation(
                x=label_x,
                y=label_y,
                text=lbl,
                showarrow=False,
                font=dict(color="black", size=10),
                xanchor="center",
                yanchor="middle",
                textangle=-text_angle
            )

    else:
        # --- Default sections (9 spokes with colored outlines and labels from notes) ---
        for v in range(1, num_vars + 1):
            angle = start + (v / (num_vars + 1)) * section_width
            angles.append(angle)

            # clockwise color ordering (as you requested earlier)
            color = colors[(num_vars - v) % len(colors)]

            # spoke line
            fig.add_trace(go.Scatter(
                x=[0, max_radius_iv * np.cos(angle)],
                y=[0, max_radius_iv * np.sin(angle)],
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=False
            ))

            # blank markers at all levels
            for lvl in range(1, num_levels + 1):
                r_lvl = max_radius_iv * (lvl / num_levels)
                x_m = r_lvl * np.cos(angle)
                y_m = r_lvl * np.sin(angle)
                fig.add_trace(go.Scatter(
                    x=[x_m], y=[y_m],
                    mode="markers",
                    marker=dict(size=4, color="white", line=dict(color=color, width=0.8)),
                    showlegend=False
                ))

            # --- Add label using annotation ---
            label_r = max_radius_iv * 1.13
            label_x = label_r * np.cos(angle)
            label_y = label_r * np.sin(angle)
            #lbl = notes[(v - 1) % len(notes)]
            lbl = notes[(num_vars - v) % len(notes)]

            text_angle = np.degrees(angle)
            if np.cos(angle) < 0:
                text_angle +=180

            fig.add_annotation(
                x=label_x,
                y=label_y,
                text=lbl,
                showarrow=False,
                font=dict(color=color, size=9),
                xanchor="center",
                yanchor="middle",
                textangle=-text_angle
            )

# --- Sidebar buttons for IV levels ---
st.sidebar.header("Rank Notes")

if "iv_levels" not in st.session_state:
    st.session_state.iv_levels = [0] * num_vars  # default mid level

for i in range(num_vars):
    note_label = notes[i]
    st.sidebar.write(note_label) #f"**Variable {i+1}**")
    cols = st.sidebar.columns(7)
    for j, col in enumerate(cols):
        if col.button(str(j), key=f"var{i}_lvl{j}"):
            st.session_state.iv_levels[i] = j


# --- Draw user polygon ---
r_points = [max_radius_iv * (lvl / num_levels) for lvl in st.session_state.iv_levels]
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
r_center = 0.15
theta_center = np.linspace(0, 2*np.pi, 200)
x_center = r_center * np.cos(theta_center)
y_center = r_center * np.sin(theta_center)
fig.add_trace(go.Scatter(x=x_center, y=y_center, fill="toself", mode="lines",
                         line=dict(color="black", width=2), fillcolor="white", showlegend=False))

fig.update_layout(
    width=700,#400,
    height=700,#400,
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

