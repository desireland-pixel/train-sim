# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
from pathlib import Path

st.set_page_config(layout="wide", page_title="Train-Warehouse Simulation")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------------
# Load CSV safely
# -------------------------
def load_csv(filename):
    path = DATA_DIR / filename
    if path.exists():
        try:
            df = pd.read_csv(path)
            if df.empty or len(df.columns) == 0:
                return pd.DataFrame()
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

trains = load_csv("trains.csv")
warehouses = load_csv("warehouses.csv")
packages = load_csv("packages.csv")
persons = load_csv("persons.csv")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Simulation Settings")
max_packages_per_person = st.sidebar.number_input("Max packages a person can carry", 1, 10, 5)
num_people = st.sidebar.number_input("If persons.csv missing, auto-create N persons", 1, 50, 10)
current_time = st.sidebar.number_input("Current time (minutes)", 0, 60, 0)

# Orders per train input
st.sidebar.markdown("### Orders per Train")
for train_id in trains.train_id:
    key = f"orders_{train_id}"
    if key not in st.session_state:
        st.session_state[key] = 0
    st.session_state[key] = st.sidebar.number_input(f"{train_id}", 0, 20, st.session_state[key])

# -------------------------
# Sample data if CSVs missing
# -------------------------
if trains.empty:
    trains = pd.DataFrame({
        'train_id': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'platform': [1, 2, 3, 4, 5],
        'start_time': [0, 10, 20, 30, 40],
        'arrive_time': [20, 30, 40, 50, 60],
        'x_source': [0, 0, 0, 0, 0],
        'y_source': [-50, -60, -70, -80, -90],
        'x_platform': [100, 200, 300, 400, 500],
        'y_platform': [0, 0, 0, 0, 0]
    })

if warehouses.empty:
    warehouses = pd.DataFrame({
        'warehouse_id': ['W1','W2','W3','W4','W5','W6'],
        'zone': ['A','A','B','B','C','C'],
        'x': [80,120,250,290,400,440],
        'y': [100,120,100,120,100,120],
        'walk_time_to_platform': [3,4,3,5,6,4]
    })

if persons.empty:
    persons = pd.DataFrame({
        'person_id': [f'Person{i+1}' for i in range(num_people)],
        'home_warehouse': np.random.choice(warehouses.warehouse_id, num_people)
    })

# -------------------------
# Generate Packages Button
# -------------------------
if st.button("Generate Packages from Orders"):
    gen_packages = []
    for i, train_id in enumerate(trains.train_id, 1):
        n_orders = st.session_state.get(f"orders_{train_id}", 0)
        if n_orders > 0:
            start_time = int(trains[trains.train_id == train_id].start_time.values[0])
            for j in range(1, n_orders+1):
                pkg_id = f"{i:02d}{j:02d}"
                warehouse_id = np.random.choice(warehouses.warehouse_id)
                gen_packages.append({
                    "package_id": pkg_id,
                    "warehouse_id": warehouse_id,
                    "generated_time": start_time - 10,
                    "quantity": 1
                })
    packages = pd.DataFrame(gen_packages)
    packages.to_csv(DATA_DIR / "packages.csv", index=False)
    st.session_state["pkg_text"] = packages[["package_id","warehouse_id","generated_time"]]

# -------------------------
# Show package text summary
# -------------------------
if "pkg_text" in st.session_state:
    st.markdown("**Generated Packages:**")
    st.dataframe(st.session_state["pkg_text"])

# -------------------------
# Digital clock
# -------------------------
base_hour = 9
total_minutes = current_time
display_hour = base_hour + total_minutes // 60
display_minute = total_minutes % 60
clock_str = f"{display_hour:02d}:{display_minute:02d}"

st.markdown(f"""
<div style='text-align: right; font-size:48px;'>
    ⏰ {clock_str}
</div>
""", unsafe_allow_html=True)

# -------------------------
# Simulation visuals (Plotly)
# -------------------------
fig = go.Figure()

# Warehouses
fig.add_trace(go.Scatter(
    x=warehouses.x, y=warehouses.y,
    mode="markers+text",
    text=warehouses.warehouse_id,
    name="Warehouses",
    marker=dict(size=15, color="green", symbol="square"),
    textposition="top center",
    textfont=dict(color="black")
))

# Platforms (5 fixed)
platforms = pd.DataFrame({
    'platform': [1,2,3,4,5],
    'x': [200,200,200,200,200],
    'y': [150,100,50,0,-50]
})

fig.add_trace(go.Scatter(
    x=platforms.x, y=platforms.y,
    mode="markers+text",
    text=[f"P{i}" for i in platforms.platform],
    name="Platforms",
    marker=dict(size=18, color="blue")
))

# Trains movement
train_positions = []
for _, r in trains.iterrows():
    if current_time < r.start_time:
        x, y = r.x_source, r.y_source
    elif current_time > r.arrive_time:
        x, y = r.x_platform, r.y_platform
    else:
        frac = (current_time - r.start_time) / (r.arrive_time - r.start_time)
        x = r.x_source + frac * (r.x_platform - r.x_source)
        y = r.y_source + frac * (r.y_platform - r.y_source)
    train_positions.append((r.train_id, x, y))

fig.add_trace(go.Scatter(
    x=[x for _,x,_ in train_positions],
    y=[y for _,_,y in train_positions],
    text=[tid for tid,_,_ in train_positions],
    mode="markers+text",
    name="Trains",
    marker=dict(size=20, color="red"),
    textfont=dict(color="white")
))

# -------------------------
# Draw packages (brown boxes) only if current_time >= generated_time
# -------------------------
if not packages.empty:
    pkg_vis = packages[packages.generated_time <= current_time]
    for wh_id in warehouses.warehouse_id:
        wh_x = warehouses.loc[warehouses.warehouse_id==wh_id, "x"].values[0]
        wh_y = warehouses.loc[warehouses.warehouse_id==wh_id, "y"].values[0]
        pkgs_here = pkg_vis[pkg_vis.warehouse_id == wh_id]
        for idx, (i, pkg) in enumerate(pkgs_here.iterrows()):
            x = wh_x + 5 + (idx % 5) * 5
            y = wh_y + (idx // 5) * 5
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=10, color="sandybrown", symbol="square"),
                text=[pkg.package_id],
                textposition="top center",
                showlegend=False
            ))

# Layout
fig.update_layout(
    title="Train, Warehouse, and Platform Map",
    xaxis=dict(title="X", range=[-50,550]),
    yaxis=dict(title="Y", range=[-100,200]),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Bottom info messages
# -------------------------
if "pkg_text" in st.session_state:
    st.success(f"Generated {len(st.session_state['pkg_text'])} packages and saved to data/packages.csv")

st.info("Use the button in the sidebar to move time forward or backward.")
