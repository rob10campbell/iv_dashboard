import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter1d
import io

st.set_page_config(layout="wide")
st.title("Coffee Cupping")

# --- Sidebar tabs ---
tab1, tab2, tab3, tab4, tab5, tab6  = st.sidebar.tabs(["Origin", "Beans", "Grind", "Wet", "Crust", "Taste"])

# --- Tab 1: Shape slider ---
#with tab1:
#    factor = st.slider("Context about origin (0 to 100)", 0, 100, 0)
#morph_strength = factor / 100
#smooth_strength = 23 / 100  # fixed fractal roughness

# --- Tab 1: Context entries instead of slider ---
context_labels = [ "Origin", "Process", "Roast", "Grower", "Grower details", 
                   "Roaster", "Roaster details", "Where purchased", "Retail details", "Other" 
]
with tab1:
    st.subheader("About this coffee")

    # --- NEW: Coffee Name input ---
    coffee_name = st.text_input("Coffee Name", key="coffee_name")
    #st.session_state.coffee_name = coffee_name  # store for access later

    st.write("Fill in up to 10 fields") # (each filled box adds 10 points)

    # Create 10 text fields
    context_inputs = []
    for i in range(10):
        value = st.text_input(context_labels[i], key=f"context_{i}")
        context_inputs.append(value)

    # Count how many are filled
    num_filled = sum(1 for v in context_inputs if v.strip() != "")
    factor = num_filled * 10  # Scale 0–10 → 0–100
    st.write(f"**Context factor:** {factor}")

# Derived parameters
morph_strength = factor / 100
smooth_strength = 23 / 100  # fixed fractal roughness


# --- Constants ---
num_points = 8000
max_radius_iv = 0.8
outer_base_radius = 1.2
theta = np.linspace(0, 2 * np.pi, num_points)

bean_colors = ["white", "#a8b448", "#dbc649", "#b58a48", "#9a6c36", "#663b15", "#311708"]
#"#a8b448", "#dbc649", "#c8a472", "#b8844e", "#9f6e48", "#652d17"]
bean_labels = ["Uniform", "Color", "Smooth", "Shiny"]
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
num_bean_vars = 4

# --- Helper: Polygon radius ---
def polygon_radius(theta, n_sides):
    return outer_base_radius / np.cos((np.pi / n_sides) - (theta % (2*np.pi / n_sides)))

# --- Base shape ---
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

# --- Adjusted section widths ---
base_angle = 2 * np.pi / num_sections
section_widths = [base_angle] * num_sections
section_widths[3] = base_angle * 0.5
section_widths[4] = base_angle * 1.5

# --- IV spokes + blank markers (GENERATE CLOCKWISE) ---
angles = []
section_indices = [[] for _ in range(num_sections)]
current_angle = 0.0  # start at 0 and move clockwise by subtracting angles

for i in range(num_sections):
    # compute the angular span for this section, but go clockwise (subtract)
    start = current_angle
    end = start - section_widths[i]   # move clockwise
    section_width = start - end       # positive magnitude

    # For masks and plotting we want monotonic arrays — create theta slice from start -> end
    if i == 3:
        # Section 3 (index 2) masked: build wedge following the outer boundary r_final
        mask_theta = np.linspace(start, end, 200)
        r_mask = np.array([np.interp((t % (2*np.pi)), theta, r_final-0.1) for t in mask_theta])
        mask_x = np.concatenate(([0.0], r_mask * np.cos(mask_theta), [0.0]))
        mask_y = np.concatenate(([0.0], r_mask * np.sin(mask_theta), [0.0]))
        fig.add_trace(go.Scatter(
            x=mask_x, y=mask_y, fill="toself", mode="lines",
            line=dict(color="white", width=0), fillcolor="white", showlegend=False
        ))
        current_angle = end
        continue

    if i == 5:
        # Section 1 custom: num_bean_vars spokes, black lines, colored first-spoke markers
        num_spokes = num_bean_vars
        for v in range(1, num_spokes + 1):
            # compute a point inside the clockwise span: fraction increases from 0..1 but we go start -> end (decreasing)
            frac = v / (num_spokes + 1)
            angle = start - frac * section_width   # decrease from start toward end
            idx = len(angles)            # index this new spoke will have in the flat array
            angles.append(angle)
            section_indices[i].append(idx)   # record that idx belongs to section i

            # spoke line (black)
            fig.add_trace(go.Scatter(
                x=[0, max_radius_iv * np.cos(angle)],
                y=[0, max_radius_iv * np.sin(angle)],
                mode="lines", line=dict(color="black", width=1.5), showlegend=False
            ))

            # markers for Section 1 — use num_levels as before (or override per your earlier logic)
            for lvl in range(1, num_levels + 1):
                r_lvl = max_radius_iv * (lvl / num_levels)
                x_m = r_lvl * np.cos(angle)
                y_m = r_lvl * np.sin(angle)

                if v == 2:  # first spoke colored markers
                    #color = bean_colors[(lvl - 1) % len(bean_colors)]
                    color = bean_colors[lvl-1]
                    bean_width = 2
                else:
                    #color = "black"
                    color = "white"
                    bean_width = 0.8

                fig.add_trace(go.Scatter(
                    x=[x_m], y=[y_m],
                    mode="markers",
                    marker=dict(size=6, color=color, line=dict(color="black", width=0.5)),
                    showlegend=False
                ))

            # label via annotation (outside the ring)
            label_r = max_radius_iv * 1.14
            label_x = label_r * np.cos(angle)
            label_y = label_r * np.sin(angle)
            #lbl = bean_labels[len(bean_labels) - v] if (len(bean_labels) - v) < len(bean_labels) else f"S1-{v}"
            lbl = bean_labels[v - 1] if (v - 1) < len(bean_labels) else f"S1-{v}"
            text_angle = np.degrees(angle)
            if np.cos(angle) < 0:
                text_angle += 180
            fig.add_annotation(
                x=label_x, y=label_y, text=lbl, showarrow=False,
                font=dict(color="black", size=12), xanchor="center", yanchor="middle",
                textangle=-text_angle
            )

        # Optionally draw Section 1 arc grid (if you want separate rings inside that slice)
        #theta_ring = np.linspace(start, end, 200)
        #for j in range(1, num_levels + 1):
        #    r_ring = max_radius_iv * (j / num_levels)
        #    fig.add_trace(go.Scatter(
        #        x=r_ring * np.cos(theta_ring),
        #        y=r_ring * np.sin(theta_ring),
        #        mode="lines",
        #        line=dict(color="black", width=0.8, dash="dot"),
        #        showlegend=False
        #    ))

        current_angle = end
        continue

    # Default other sections (clockwise)
    for v in range(1, num_vars + 1):
        frac = v / (num_vars + 1)
        angle = start - frac * section_width
        idx = len(angles)            # index this new spoke will have in the flat array
        angles.append(angle)
        section_indices[i].append(idx)   # record that idx belongs to section i
        color = colors[(v - 1) % len(colors)]
        #color = colors[(num_vars - v) % len(colors)]

        # spoke line
        fig.add_trace(go.Scatter(
            x=[0, max_radius_iv * np.cos(angle)],
            y=[0, max_radius_iv * np.sin(angle)],
            mode="lines", line=dict(color=color, width=1), showlegend=False
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

        # label (annotation)
        label_r = max_radius_iv * 1.14
        label_x = label_r * np.cos(angle)
        label_y = label_r * np.sin(angle)
        lbl = notes[(v - 1) % len(notes)]
        #lbl = notes[(num_vars - v) % len(notes)]
        text_angle = np.degrees(angle)
        if np.cos(angle) < 0:
            text_angle += 180
        fig.add_annotation(
            x=label_x, y=label_y, text=lbl, showarrow=False,
            font=dict(color=color, size=12),
            xanchor="center", yanchor="middle", textangle=-text_angle
        )

    current_angle = end  # next section continues clockwise


# set up user ranking
total_spokes = len(angles)
if "iv_levels" not in st.session_state:
    st.session_state.iv_levels = [1] * total_spokes

# --- CSS for dots ---
st.markdown("""
<style>
.dot {
    height: 8px;
    width: 8px;
    background-color: #bbb;
    border-radius: 50%;
    display: inline-block;
    margin-top: 4px;
    transition: background-color 0.3s;
}
.dot.active {
    background-color: #4CAF50;
}
.dot-row {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# --- Tab 2: Rank Beans ---
with tab2:
    st.header("Step 1: Beans")
    sec = 5  # which section in section_indices corresponds to "Beans"

    for local_i, global_idx in enumerate(section_indices[sec]):
        bean_label = bean_labels[local_i]
        st.write(bean_label)

        # --- 1️⃣ Draw dot row FIRST (visually above buttons)
        current_level = st.session_state.iv_levels[global_idx]
        dot_cols = st.columns(7)
        for j, dcol in enumerate(dot_cols):
            active = (current_level == j + 1)
            dcol.markdown(
                f"""
                <div style="text-align:center;">
                  <span style="
                    display:inline-block;
                    width:10px; height:10px;
                    border-radius:50%;
                    background-color: {'#4CAF50' if active else '#bbb'};
                    margin-bottom:6px;"></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- 2️⃣ Then draw buttons SECOND (so they're below visually)
        cols = st.columns(7)
        for j, col in enumerate(cols):
            key = f"bean_sec{sec}_var{local_i}_lvl{j+1}"
            if col.button(str(j), key=key):
                # Update session state when button is clicked
                st.session_state.iv_levels[global_idx] = j + 1
                st.rerun()  # <--- Force refresh so dots update immediately

# --- Tab 3: Rank Grind ---
with tab3:
    st.header("Step 2: Aroma (grind)")
    sec = 0

    for local_i, global_idx in enumerate(section_indices[sec]):
        note_label = notes[local_i]
        var_color = colors[local_i]

        # --- Label with color
        st.markdown(
            f"<p style='color:{var_color}; font-weight:bold;'>{note_label}</p>",
            unsafe_allow_html=True
        )

        # --- 1️⃣ Draw dot row FIRST (visually above buttons)
        current_level = st.session_state.iv_levels[global_idx]
        dot_cols = st.columns(7)
        for j, dcol in enumerate(dot_cols):
            active = (current_level == j + 1)
            dcol.markdown(
                f"""
                <div style="text-align:center;">
                  <span style="
                    display:inline-block;
                    width:10px; height:10px;
                    border-radius:50%;
                    background-color: {var_color if active else '#bbb'};
                    margin-bottom:6px;"></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- 2️⃣ Then draw buttons SECOND (so they're below visually)
        cols = st.columns(7)
        for j, col in enumerate(cols):
            key = f"grind_sec{sec}_var{local_i}_lvl{j+1}"
            if col.button(str(j), key=key):
                # Update session state when button is clicked
                st.session_state.iv_levels[global_idx] = j + 1
                st.rerun()  # <--- Force refresh so dots update immediately




# --- Tab 4: Rank Wetted ---
with tab4:
    st.header("Step 3: Aroma (wet)")
    sec = 1

    for local_i, global_idx in enumerate(section_indices[sec]):
        note_label = notes[local_i]
        var_color = colors[local_i]

        # --- Label with color
        st.markdown(
            f"<p style='color:{var_color}; font-weight:bold;'>{note_label}</p>",
            unsafe_allow_html=True
        )

        # --- 1️⃣ Draw dot row FIRST (visually above buttons)
        current_level = st.session_state.iv_levels[global_idx]
        dot_cols = st.columns(7)
        for j, dcol in enumerate(dot_cols):
            active = (current_level == j + 1)
            dcol.markdown(
                f"""
                <div style="text-align:center;">
                  <span style="
                    display:inline-block;
                    width:10px; height:10px;
                    border-radius:50%;
                    background-color: {var_color if active else '#bbb'};
                    margin-bottom:6px;"></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- 2️⃣ Then draw buttons SECOND (so they're below visually)
        cols = st.columns(7)
        for j, col in enumerate(cols):
            key = f"wet_sec{sec}_var{local_i}_lvl{j+1}"
            if col.button(str(j), key=key):
                # Update session state when button is clicked
                st.session_state.iv_levels[global_idx] = j + 1
                st.rerun()  # <--- Force refresh so dots update immediately


# --- Tab 5: Rank Crust ---
with tab5:
    st.header("Step 4: Aroma (crust)")
    sec = 2

    for local_i, global_idx in enumerate(section_indices[sec]):
        note_label = notes[local_i]
        var_color = colors[local_i]

        # --- Label with color
        st.markdown(
            f"<p style='color:{var_color}; font-weight:bold;'>{note_label}</p>",
            unsafe_allow_html=True
        )

        # --- 1️⃣ Draw dot row FIRST (visually above buttons)
        current_level = st.session_state.iv_levels[global_idx]
        dot_cols = st.columns(7)
        for j, dcol in enumerate(dot_cols):
            active = (current_level == j + 1)
            dcol.markdown(
                f"""
                <div style="text-align:center;">
                  <span style="
                    display:inline-block;
                    width:10px; height:10px;
                    border-radius:50%;
                    background-color: {var_color if active else '#bbb'};
                    margin-bottom:6px;"></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- 2️⃣ Then draw buttons SECOND (so they're below visually)
        cols = st.columns(7)
        for j, col in enumerate(cols):
            key = f"crust_sec{sec}_var{local_i}_lvl{j+1}"
            if col.button(str(j), key=key):
                # Update session state when button is clicked
                st.session_state.iv_levels[global_idx] = j + 1
                st.rerun()  # <--- Force refresh so dots update immediately

# --- Tab 6: Rank Taste ---
with tab6:
    st.header("Step 6: Taste")
    sec = 4

    for local_i, global_idx in enumerate(section_indices[sec]):
        note_label = notes[local_i]
        var_color = colors[local_i]

        # --- Label with color
        st.markdown(
            f"<p style='color:{var_color}; font-weight:bold;'>{note_label}</p>",
            unsafe_allow_html=True
        )

        # --- 1️⃣ Draw dot row FIRST (visually above buttons)
        current_level = st.session_state.iv_levels[global_idx]
        dot_cols = st.columns(7)
        for j, dcol in enumerate(dot_cols):
            active = (current_level == j + 1)
            dcol.markdown(
                f"""
                <div style="text-align:center;">
                  <span style="
                    display:inline-block;
                    width:10px; height:10px;
                    border-radius:50%;
                    background-color: {var_color if active else '#bbb'};
                    margin-bottom:6px;"></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- 2️⃣ Then draw buttons SECOND (so they're below visually)
        cols = st.columns(7)
        for j, col in enumerate(cols):
            key = f"grind_sec{sec}_var{local_i}_lvl{j+1}"
            if col.button(str(j), key=key):
                # Update session state when button is clicked
                st.session_state.iv_levels[global_idx] = j + 1
                st.rerun()  # <--- Force refresh so dots update immediately



# --- Draw user polygon ---
r_points = [max_radius_iv * (lvl / num_levels) for lvl in st.session_state.iv_levels]
# sanity check
assert len(r_points) == len(angles)

x_poly = [r_points[i] * np.cos(angles[i]) for i in range(len(angles))]
y_poly = [r_points[i] * np.sin(angles[i]) for i in range(len(angles))]
x_poly.append(x_poly[0]); y_poly.append(y_poly[0])


fig.add_trace(go.Scatter(
    x=x_poly, y=y_poly,
    mode="lines+markers", fill="toself",
    line=dict(color="black", width=3),
    fillcolor="rgba(0,0,0,0.2)",
    marker=dict(size=6, color="black"),
    name="User Selection"
))

# --- Section lines ---
for i in range(num_sections):
    if i == 5:
        angle = 0.5 * i * (2 * np.pi / num_sections)
        idx = (np.abs(theta - angle)).argmin()
        r_edge = r_final[idx]
        fig.add_trace(go.Scatter(
            x=[0, r_edge*np.cos(angle)], y=[0, r_edge*np.sin(angle)],
            mode="lines", line=dict(color="black", width=2), showlegend=False))
    if i != 2:
        angle = i * (2 * np.pi / num_sections)
        idx = (np.abs(theta - angle)).argmin()
        r_edge = r_final[idx]
        fig.add_trace(go.Scatter(
            x=[0, r_edge*np.cos(angle)], y=[0, r_edge*np.sin(angle)],
            mode="lines", line=dict(color="black", width=2), showlegend=False))



# --- Center cover circle ---
r_center = 0.15
theta_center = np.linspace(0, 2*np.pi, 200)
x_center = r_center * np.cos(theta_center)
y_center = r_center * np.sin(theta_center)
fig.add_trace(go.Scatter(
    x=x_center, y=y_center, fill="toself", mode="lines",
    line=dict(color="black", width=2), fillcolor="white", showlegend=False))

# --- Display coffee name in the center ---
# --- Safely read the coffee name from session_state (if it exists) ---
coffee_name = st.session_state.get("coffee_name", "")
coffee_name = coffee_name.strip() if coffee_name else ""

if coffee_name:
    # --- Auto-insert line breaks every ~15 characters between words ---
    words = coffee_name.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= 8:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    wrapped_name = "<br>".join(lines)

    # --- Add wrapped name annotation ---
    fig.add_annotation(
        x=0, y=0,
        text=wrapped_name,
        showarrow=False,
        font=dict(size=11, color="black", family="Arial Black"),
        xanchor="center", yanchor="middle"
    )



# --- Final image size ---
fig.update_layout(
    width=700, height=700,
    xaxis=dict(scaleanchor="y", visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="white",
)



# --- Sidebar formatting ---
st.markdown("""
    <style>
        /* Make the sidebar wider */
        [data-testid="stSidebar"] {
            width: 400px;       /* default is around 250px */
        }
    </style>
""", unsafe_allow_html=True)


# --- ICONS ---
section_icons = [
    "https://raw.githubusercontent.com/rob10campbell/iv_dashboard/main/icons/2grind.png",
    "https://raw.githubusercontent.com/rob10campbell/iv_dashboard/main/icons/3wet.png", 
    "https://raw.githubusercontent.com/rob10campbell/iv_dashboard/main/icons/4crust.png",
    "https://raw.githubusercontent.com/rob10campbell/iv_dashboard/main/icons/5clearwait.png",
    #"https://raw.githubusercontent.com/rob10campbell/iv_dashboard/main/icons/6wait.png", 
    "https://raw.githubusercontent.com/rob10campbell/iv_dashboard/main/icons/7slurp.png",
    "https://raw.githubusercontent.com/rob10campbell/iv_dashboard/main/icons/8chat.png",
    "https://raw.githubusercontent.com/rob10campbell/iv_dashboard/main/icons/1bean.png"
]


for i, icon_path in enumerate(section_icons):
    if i == 0: # grind
      icon_radius = 1.16 
      theta_mid = -i * (2 * np.pi / num_sections) - (base_angle / 2)
      size_x = 0.28
      size_y = 0.28
    elif i == 1: # wet
      icon_radius = 1.1
      theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * 0.25) # bottom
      size_x = 0.32
      size_y = 0.32
    elif i == 2: # crust
      icon_radius = 1.23
      theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * 0.35) # top
      size_x = 0.43
      size_y = 0.43
    elif i == 3: # clear+wait
      icon_radius = 1  # slightly outside outer_base_radius
      #theta_mid = -i * (2 * np.pi / num_sections) - (base_angle / 2)
      #theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * 0.75) # bottom
      theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * 0.25) # top
      size_x = 0.43
      size_y = 0.43
    #elif i == 4:
    #  icon_radius = 1.05
    #  #theta_mid = -i * (2 * np.pi / num_sections) - (base_angle / 2)
    #  #theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * 0.75) # bottom
    #  theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * -0.50) # top
    #  size_x = 0.25
    #  size_y = 0.25
    elif i == 4: # slurp
      icon_radius = 1.2  # slightly outside outer_base_radius
      #theta_mid = -i * (2 * np.pi / num_sections) - (base_angle / 2)
      #theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * 0.75) # bottom
      theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * -0.3) # top
      size_x = 0.33
      size_y = 0.33
    elif i == 5: # chat
      icon_radius = 1.11  # slightly outside outer_base_radius
      #theta_mid = -i * (2 * np.pi / num_sections) - (base_angle / 2)
      #theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * 0.75) # bottom
      theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * -0.3) # top
      size_x = 0.24
      size_y = 0.24
    elif i == 6: # bean
      icon_radius = 1.18  # slightly outside outer_base_radius
      #theta_mid = -i * (2 * np.pi / num_sections) - (base_angle / 2)
      #theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * 0.75) # bottom
      theta_mid = -i * (2 * np.pi / num_sections) - (base_angle * -0.7) # top
      size_x = 0.18
      size_y = 0.18
    x_icon = icon_radius * np.cos(theta_mid)
    y_icon = icon_radius * np.sin(theta_mid)

    fig.add_layout_image(
        dict(
            source=icon_path,
            xref="x",
            yref="y",
            x=x_icon,
            y=y_icon,
            sizex=size_x,  # adjust for icon scale
            sizey=size_y,
            xanchor="center",
            yanchor="middle",
            layer="above"
        )
    )


st.plotly_chart(fig, width='stretch') # False for left-adjusted, True for centered


# --- Download Button ---
img_bytes = fig.to_image(format="png", width=1000, height=1000, scale=2)
st.download_button("📥 Download current image", img_bytes, "tasting_diagram.png", "image/png")

