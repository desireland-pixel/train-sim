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
# Load data (CSVs)
# -------------------------
def load_csv(filename):
    path = DATA_DIR / filename
    if path.exists():
        try:
            df = pd.read_csv(path)
            # if file exists but has no rows or columns
            if df.empty or len(df.columns) == 0:
                return pd.DataFrame()
            return df
        except pd.errors.EmptyDataError:
            # happens when file is 0 bytes (empty)
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def generate_packages_from_orders(orders_list, trains_df, warehouses_df):
    """
    orders_list: list of 5 integers -> orders for T1..T5
    trains_df: DataFrame with trains; must include columns 'train_id' and 'start_time'
    warehouses_df: DataFrame with warehouses; must include 'warehouse_id'
    Returns DataFrame with columns: package_id, warehouse_id, generated_time, quantity
    """
    rows = []
    # ensure trains are in order T1..T5; we'll map train index 1..5 to trains rows
    # If trains_df has train order different, we'll pick by platform or index order; here we use trains_df sorted by train_id
    trains_sorted = trains_df.copy().reset_index(drop=True)
    # If trains_sorted has fewer than 5 rows, rely on whatever is present
    for t_index, num_orders in enumerate(orders_list, start=1):
        if num_orders <= 0:
            continue
        # find corresponding train row (prefer train with train_id T{t_index} if exists)
        # fallback to index-based selection
        train_row = None
        tid = f"T{t_index}"
        if 'train_id' in trains_sorted.columns and tid in trains_sorted['train_id'].values:
            train_row = trains_sorted[trains_sorted['train_id'] == tid].iloc[0]
        elif len(trains_sorted) >= t_index:
            train_row = trains_sorted.iloc[t_index - 1]
        else:
            # if no train info available, default start_time to 30
            train_row = {'start_time': 30}
        start_time = int(train_row.get('start_time', 30))
        generated_time = max(0, start_time - 10)  # start_time - 10 minutes, not negative

        # generate packages for this train
        for i in range(1, num_orders + 1):
            package_id = f"{t_index:02d}{i:02d}"    # TTNN format: e.g. 0101
            # random warehouse selection
            if 'warehouse_id' in warehouses_df.columns and len(warehouses_df) > 0:
                wh = random.choice(list(warehouses_df['warehouse_id']))
            else:
                wh = 'W1'
            rows.append({
                'package_id': package_id,
                'warehouse_id': wh,
                'generated_time': generated_time,
                'quantity': 1
            })
    return pd.DataFrame(rows)

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

st.sidebar.markdown("---")
st.sidebar.markdown("### Orders per Train (manual)")

orders_T1 = st.sidebar.number_input("Orders for T1", min_value=0, max_value=50, value=0, key="ord_t1")
orders_T2 = st.sidebar.number_input("Orders for T2", min_value=0, max_value=50, value=0, key="ord_t2")
orders_T3 = st.sidebar.number_input("Orders for T3", min_value=0, max_value=50, value=0, key="ord_t3")
orders_T4 = st.sidebar.number_input("Orders for T4", min_value=0, max_value=50, value=0, key="ord_t4")
orders_T5 = st.sidebar.number_input("Orders for T5", min_value=0, max_value=50, value=0, key="ord_t5")

if st.sidebar.button("Generate Packages from Orders"):
    # we'll call the generator (function added later) and overwrite data/packages.csv
    orders = [int(orders_T1), int(orders_T2), int(orders_T3), int(orders_T4), int(orders_T5)]
    gen_df = generate_packages_from_orders(orders, trains, warehouses)
    # save to CSV
    gen_df.to_csv(DATA_DIR / "packages.csv", index=False)
    st.success(f"Generated {len(gen_df)} packages and saved to data/packages.csv")
    # make Streamlit reload the variable (optional): re-read packages
    packages = gen_df.copy()

# -------------------------
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
    # no packages.csv found -> show empty DataFrame and let user generate using the sidebar
    packages = pd.DataFrame(columns=['package_id', 'warehouse_id', 'generated_time', 'quantity'])

if persons.empty:
    persons = pd.DataFrame({
        'person_id': [f'Person{i+1}' for i in range(num_people)],
        'home_warehouse': np.random.choice(warehouses.warehouse_id, num_people)
    })

# -------------------------
# Simulation visuals
# -------------------------
fig = go.Figure()

# Compute digital clock
base_hour = 9
base_minute = 0
total_minutes = base_minute + current_time
display_hour = base_hour + total_minutes // 60
display_minute = total_minutes % 60
clock_str = f"{display_hour:02d}:{display_minute:02d}"

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

# --- display packages as small light-brown squares to the right of each warehouse ---
if not packages.empty:
    # expand packages into list if quantity >1 (your generator sets quantity=1)
    pk = packages.copy().reset_index(drop=True)

    # Build a map: warehouse_id -> list of packages (order preserved)
    wh_groups = {}
    for idx, row in pk.iterrows():
        wid = row['warehouse_id']
        if wid not in wh_groups:
            wh_groups[wid] = []
        wh_groups[wid].append(row['package_id'])

    # For each package compute an x,y offset to the right of the warehouse
    pkg_x = []
    pkg_y = []
    pkg_text = []
    # layout parameters for stacking
    x_offset_base = 12     # horizontal offset to the right of warehouse
    col_spacing = 12       # horizontal separation between package columns
    row_spacing = 10       # vertical separation between rows
    max_cols = 6           # max columns before wrapping to next row

    wh_coords = {row['warehouse_id']:(row['x'], row['y']) for _, row in warehouses.iterrows()}

    for wh_id, pkg_list in wh_groups.items():
        wx, wy = wh_coords.get(wh_id, (0,0))
        for j, pid in enumerate(pkg_list):
            col = j % max_cols
            rown = j // max_cols
            x_pos = wx + x_offset_base + col * col_spacing
            y_pos = wy - rown * row_spacing
            pkg_x.append(x_pos)
            pkg_y.append(y_pos)
            pkg_text.append(pid)

    # add packages as a single trace
    fig.add_trace(go.Scatter(
        x=pkg_x,
        y=pkg_y,
        mode="markers+text",
        text=pkg_text,
        textposition="middle right",
        name="Packages",
        marker=dict(size=12, color="#D2B48C", symbol="square", line=dict(color="black", width=0.5)),
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
    marker=dict(size=20, color="red"),
    textfont=dict(color="white")
))

fig.update_layout(
    title="Train, Warehouse, and Platform Map",
    xaxis=dict(title="X", range=[-50,550]),
    yaxis=dict(title="Y", range=[-100,200]),
    height=600
)

# Show clock
st.markdown(f"""
<div style='text-align: right; font-size:48px;'>
    ⏰ {clock_str}
</div>
""", unsafe_allow_html=True)

# Show chart
st.plotly_chart(fig, use_container_width=True)

st.info("Use the button in the sidebar to move time forward or backward.")
