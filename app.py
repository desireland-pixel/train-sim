# app.py
import streamlit as st
import pandas as pd
import numpy as np
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

current_time = st.sidebar.slider("Current time (minutes)", 0, 120, 0)

st.title("🚉 Train–Warehouse Simulation")
st.markdown(f"**Simulation Time: {current_time} min**")

# -------------------------
# If files missing, create sample data
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

if packages.empty:
    packages = pd.DataFrame({
        'package_id': [f'P{i+1}' for i in range(10)],
        'warehouse_id': np.random.choice(warehouses.warehouse_id, 10),
        'generated_time': np.random.randint(0, 60, 10),
        'quantity': np.random.randint(1, 5, 10)
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
    marker=dict(size=15, color="green")
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

# Trains movement (linear between source and platform)
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
    marker=dict(size=20, color="red")
))

fig.update_layout(
    title="Train, Warehouse, and Platform Map",
    xaxis=dict(title="X", range=[-50,550]),
    yaxis=dict(title="Y", range=[-100,200]),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

st.info("Use the slider in the sidebar to move time forward or backward.")
