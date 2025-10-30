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
# Load CSV files
# -------------------------
def load_csv(filename):
    path = DATA_DIR / filename
    if path.exists():
        return pd.read_csv(path)
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

st.sidebar.markdown("### Orders per Train")
orders_T = {}
for t in ['T1','T2','T3','T4','T5']:
    orders_T[t] = st.sidebar.number_input(f"Orders {t}", 0, 20, 0)

generate_btn = st.sidebar.button("Generate Packages from Orders")

# Current time
current_time = st.sidebar.number_input("Current time (minutes)", 0, 60, 0)

# -------------------------
# Page Title
# -------------------------
st.title("🚉 Train–Warehouse Simulation")
st.markdown(f"**Simulation Time: {current_time} min**")

# -------------------------
# Sample Data (if CSV missing)
# -------------------------
if trains.empty:
    trains = pd.DataFrame({
        'train_id': ['T1','T2','T3','T4','T5'],
        'platform': [1,2,3,4,5],
        'start_time': [0,10,20,30,40],
        'arrive_time': [20,30,40,50,60],
        'x_source': [0,0,0,0,0],
        'y_source': [-50,-60,-70,-80,-90],
        'x_platform': [100,200,300,400,500],
        'y_platform': [0,0,0,0,0]
    })

if warehouses.empty:
    warehouses = pd.DataFrame({
        'warehouse_id': ['W1','W2','W3','W4','W5','W6'],
        'zone': ['A','A','B','B','C','C'],
        'x': [80,120,250,290,400,440],
        'y': [100,120,100,120,100,120],
        'walk_time_to_platform': [3,4,3,5,6,4]
    })

if packages.empty:
    packages = pd.DataFrame(columns=['package_id','warehouse_id','generated_time','quantity'])

if persons.empty:
    persons = pd.DataFrame({
        'person_id':[f'Person{i+1}' for i in range(num_people)],
        'home_warehouse': np.random.choice(warehouses.warehouse_id,num_people)
    })

# -------------------------
# Generate Packages from Orders
# -------------------------
generated_pkg_text = None

if generate_btn:
    new_packages = []
    for t_idx, t in enumerate(['T1','T2','T3','T4','T5'], start=1):
        n_orders = orders_T[t]
        if n_orders <= 0:
            continue
        train_row = trains[trains.train_id==t].iloc[0]
        gen_time = train_row.start_time - 10  # 10 min before train
        for i in range(1, n_orders+1):
            pkg_id = f"{t_idx:02d}{i:02d}"
            wh = np.random.choice(warehouses.warehouse_id)
            new_packages.append({'package_id':pkg_id,
                                 'warehouse_id':wh,
                                 'generated_time':gen_time,
                                 'quantity':1})
    if new_packages:
        new_pkgs_df = pd.DataFrame(new_packages)
        # Save to packages.csv
        packages = pd.concat([packages,new_pkgs_df],ignore_index=True)
        packages.to_csv(DATA_DIR / "packages.csv", index=False)
        # Create a display table with HH:MM
        display_df = new_pkgs_df.copy()
        display_df['HH:MM'] = display_df['generated_time'].apply(lambda x: f"09:{int(x):02d}")
        generated_pkg_text = display_df[['package_id','warehouse_id','HH:MM']]
        st.success(f"Generated {len(new_packages)} packages and saved to data/packages.csv")

# -------------------------
# Simulation visuals
# -------------------------
fig = go.Figure()

# Clock at top-right
base_hour = 9
total_minutes = current_time
display_hour = base_hour + total_minutes // 60
display_minute = total_minutes % 60
clock_str = f"{display_hour:02d}:{display_minute:02d}"

st.markdown(f"""
<div style='text-align:right; font-size:48px;'>
    ⏰ {clock_str}
</div>
""", unsafe_allow_html=True)

# Warehouses
fig.add_trace(go.Scatter(
    x=warehouses.x,
    y=warehouses.y,
    mode="markers+text",
    text=warehouses.warehouse_id,
    name="Warehouses",
    marker=dict(size=15, color="green", symbol="square"),
    textposition="top center",
    textfont=dict(color="black")
))

# Platforms (5 fixed)
platforms = pd.DataFrame({
    'platform':[1,2,3,4,5],
    'x':[200,200,200,200,200],
    'y':[150,100,50,0,-50]
})
fig.add_trace(go.Scatter(
    x=platforms.x,
    y=platforms.y,
    mode="markers+text",
    text=[f"P{i}" for i in platforms.platform],
    name="Platforms",
    marker=dict(size=18, color="blue")
))

# Trains
train_positions = []
for _, r in trains.iterrows():
    if current_time < r.start_time:
        x,y = r.x_source, r.y_source
    elif current_time > r.arrive_time:
        x,y = r.x_platform, r.y_platform
    else:
        frac = (current_time - r.start_time)/(r.arrive_time - r.start_time)
        x = r.x_source + frac*(r.x_platform - r.x_source)
        y = r.y_source + frac*(r.y_platform - r.y_source)
    train_positions.append((r.train_id,x,y))

fig.add_trace(go.Scatter(
    x=[x for _,x,_ in train_positions],
    y=[y for _,_,y in train_positions],
    text=[tid for tid,_,_ in train_positions],
    mode="markers+text",
    name="Trains",
    marker=dict(size=20, color="red"),
    textfont=dict(color="white")
))

# Packages: show only if current_time >= generated_time
pkg_vis = packages[packages.generated_time <= current_time]
pkg_offsets = {wh:0 for wh in warehouses.warehouse_id}
pkg_xs, pkg_ys, pkg_texts = [], [], []
for _, pkg in pkg_vis.iterrows():
    wh = pkg.warehouse_id
    wh_row = warehouses[warehouses.warehouse_id==wh].iloc[0]
    offset = pkg_offsets[wh]*10
    pkg_offsets[wh] += 1
    pkg_xs.append(wh_row.x + 10 + offset)
    pkg_ys.append(wh_row.y)
    pkg_texts.append(pkg.package_id)

if len(pkg_xs) > 0:
    fig.add_trace(go.Scatter(
        x=pkg_xs,
        y=pkg_ys,
        mode="markers+text",
        text=pkg_texts,
        name="Packages",
        marker=dict(size=10, color="peru", symbol="square"),
        textposition="top center",
        textfont=dict(size=10)
    ))

fig.update_layout(
    title="Train, Warehouse, and Platform Map",
    xaxis=dict(title="X", range=[-50,550]),
    yaxis=dict(title="Y", range=[-100,200]),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Show generated packages table (immediately after clicking button)
if generated_pkg_text is not None:
    st.markdown("### Generated Packages")
    st.dataframe(generated_pkg_text)

st.info("Use the button in the sidebar to move time forward or backward.")
