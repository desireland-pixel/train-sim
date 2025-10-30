# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(layout="wide", page_title="Train-Warehouse Simulation")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------------
# Load data (CSVs)
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

# -------------------------
# Orders per train inputs
# -------------------------
st.sidebar.markdown("### Orders per Train")
order_T1 = st.sidebar.number_input("T1 Orders", 0, 20, 0)
order_T2 = st.sidebar.number_input("T2 Orders", 0, 20, 0)
order_T3 = st.sidebar.number_input("T3 Orders", 0, 20, 0)
order_T4 = st.sidebar.number_input("T4 Orders", 0, 20, 0)
order_T5 = st.sidebar.number_input("T5 Orders", 0, 20, 0)
train_orders = [order_T1, order_T2, order_T3, order_T4, order_T5]

# -------------------------
# Generate Packages Button
# -------------------------
if st.sidebar.button("Generate Packages from Orders"):
    gen_packages = []

    # For each train, use the manual order inputs from sidebar
    for i, train_id in enumerate(trains.train_id, 1):
        n_orders = train_orders[i - 1]  # take order count directly from sidebar
        if n_orders > 0:
            start_time = int(trains.loc[trains.train_id == train_id, "start_time"].values[0])
            for j in range(1, n_orders + 1):
                pkg_id = f"{i:02d}{j:02d}"  # 0101, 0102 ... etc.
                warehouse_id = np.random.choice(warehouses.warehouse_id)  # random W1–W6
                gen_packages.append({
                    "package_id": pkg_id,
                    "warehouse_id": warehouse_id,
                    "generated_time": start_time - 10
                })

    # Convert only if we actually generated packages
    if gen_packages:
        packages = pd.DataFrame(gen_packages)
        st.session_state["packages"] = packages
        packages.to_csv(DATA_DIR / "packages.csv", index=False)
        st.session_state["pkg_text"] = packages[["package_id", "warehouse_id", "generated_time"]]
    else:
        st.warning("No orders entered — no packages generated.")
        st.session_state.pop("packages", None)
        st.session_state.pop("pkg_text", None)

# -------------------------
# Show package text summary
# -------------------------
if "pkg_text" in st.session_state:
    st.markdown("**Generated Packages:**")
    st.dataframe(st.session_state["pkg_text"])

# -------------------------
# Page title
# -------------------------
st.title("🚉 Train–Warehouse Simulation")
st.markdown(f"**Simulation Time: {current_time} min**")

# -------------------------
# Fallback data if CSVs missing
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
# Simulation visuals
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
# Packages display
# -------------------------
base_hour = 9
base_minute = 0

if "packages" in st.session_state:
    packages = st.session_state["packages"]
    
    for _, pkg in packages.iterrows():
        if current_time >= pkg.generated_time:
            wh = warehouses[warehouses.warehouse_id == pkg.warehouse_id].iloc[0]
            # position slightly to right of warehouse
            x = wh.x + 5 + (int(pkg.package_id[-2:])-1)*5
            y = wh.y
            clock_str = f"{base_hour + pkg.generated_time//60:02d}:{base_minute + pkg.generated_time%60:02d}"
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                text=[pkg.package_id],
                textposition="top center",
                marker=dict(size=12, color="#D2B48C", symbol="square"),
                name=f"Package {pkg.package_id}"
            ))

# -------------------------
# Clock on top right
# -------------------------
total_minutes = base_minute + current_time
display_hour = base_hour + total_minutes // 60
display_minute = total_minutes % 60
clock_str = f"{display_hour:02d}:{display_minute:02d}"
st.markdown(f"""
<div style='text-align: right; font-size:48px;'>
    ⏰ {clock_str}
</div>
""", unsafe_allow_html=True)

# Show chart
st.plotly_chart(fig, use_container_width=True)

st.info("Use the button in the sidebar to move time forward or backward.")
