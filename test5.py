import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="FleetRoute AI | VRP & Fuel Optimizer",
    page_icon="🚚",
    layout="wide",
)


# ==========================================
# 1. ฟังก์ชันคลีนข้อมูล & ตรรกะภูมิศาสตร์
# ==========================================
def clean_demand(val):
  if pd.isna(val):
    return 0.0
  if isinstance(val, (int, float)):
    return float(val)
  s = str(val).replace("kg", "").replace("Kg", "").replace(",", "").strip()
  try:
    return float(s)
  except:
    return 0.0


def assign_zone(lat, lon, node_id):
  if node_id == 0:
    return "Depot (ศูนย์กระจายสินค้า)"
  if lat >= 17.0:
    return "โซนภาคเหนือ"
  elif lon >= 101.5 and lat >= 14.3:
    return "โซนภาคตะวันออกเฉียงเหนือ"
  elif lon >= 100.8 and lat < 14.2:
    return "โซนภาคตะวันออก"
  elif (13.4 <= lat <= 14.1) and (100.3 <= lon <= 100.9):
    return "โซนกรุงเทพฯ และปริมณฑล"
  else:
    return "โซนภาคกลางและตะวันตก"


def haversine_distance(lat1, lon1, lat2, lon2):
  R = 6371.0
  dlat = np.radians(lat2 - lat1)
  dlon = np.radians(lon2 - lon1)
  a = (
      np.sin(dlat / 2.0) ** 2
      + np.cos(np.radians(lat1))
      * np.cos(np.radians(lat2))
      * np.sin(dlon / 2.0) ** 2
  )
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  return R * c


def get_distance_matrix(sub_df):
  n = len(sub_df)
  lats = sub_df["Latitude"].values
  lons = sub_df["Longitude"].values
  matrix = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      if i != j:
        matrix[i][j] = haversine_distance(lats[i], lons[i], lats[j], lons[j])
  return matrix


def calculate_route_distance(route, dist_matrix):
  total = 0.0
  for i in range(len(route) - 1):
    total += dist_matrix[route[i]][route[i + 1]]
  return total


def make_google_maps_link(sub_df, route):
  coords = [
      f"{sub_df.iloc[node]['Latitude']},{sub_df.iloc[node]['Longitude']}"
      for node in route
  ]
  return f"https://www.google.com/maps/dir/{'/'.join(coords)}"


def calculate_fuel_and_truck(distance, load, cap_4w, cap_6w, is_urban):
  if load <= cap_4w:
    truck_type = "รถ 4 ล้อ"
    km_per_liter = 7.25 if is_urban else 9.25
  else:
    truck_type = "รถ 6 ล้อ"
    km_per_liter = 5.25 if is_urban else 7.00
  fuel_liters = distance / km_per_liter if km_per_liter > 0 else 0
  return truck_type, km_per_liter, fuel_liters


# ==========================================
# 2. VRP Algorithms
# ==========================================
def solve_clarke_wright(dist_matrix, demands, capacity):
  n = len(demands)
  savings = []
  for i in range(1, n):
    for j in range(i + 1, n):
      s = dist_matrix[0][i] + dist_matrix[0][j] - dist_matrix[i][j]
      savings.append((s, i, j))

  savings.sort(reverse=True, key=lambda x: x[0])
  routes = {i: [0, i, 0] for i in range(1, n)}
  route_loads = {i: demands[i] for i in range(1, n)}

  for s, i, j in savings:
    route_i = None
    route_j = None
    for r_id, r in routes.items():
      if i in r:
        route_i = r_id
      if j in r:
        route_j = r_id

    if route_i is not None and route_j is not None and route_i != route_j:
      r1 = routes[route_i]
      r2 = routes[route_j]
      if route_loads[route_i] + route_loads[route_j] <= capacity:
        if r1[-2] == i and r2[1] == j:
          new_route = r1[:-1] + r2[1:]
          routes[route_i] = new_route
          route_loads[route_i] += route_loads[route_j]
          del routes[route_j]
          del route_loads[route_j]
        elif r2[-2] == j and r1[1] == i:
          new_route = r2[:-1] + r1[1:]
          routes[route_i] = new_route
          route_loads[route_i] += route_loads[route_j]
          del routes[route_j]
          del route_loads[route_j]

  return list(routes.values())


def solve_nearest_neighbor(dist_matrix, demands, capacity):
  unvisited = set(range(1, len(demands)))
  routes = []
  while unvisited:
    current_route = [0]
    current_load = 0.0
    current_node = 0
    while True:
      feasible_nodes = [
          node
          for node in unvisited
          if current_load + demands[node] <= capacity
      ]
      if not feasible_nodes:
        break
      next_node = min(
          feasible_nodes, key=lambda node: dist_matrix[current_node][node]
      )
      current_route.append(next_node)
      current_load += demands[next_node]
      unvisited.remove(next_node)
      current_node = next_node
    current_route.append(0)
    routes.append(current_route)
  return routes


# ==========================================
# 3. แผนที่ Clean Map
# ==========================================
def plot_clean_map(all_results_df, df_original):
  fig = go.Figure()
  ScatterTrace = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))
  map_layer_key = "map" if hasattr(go, "Scattermap") else "mapbox"

  palette = px.colors.qualitative.Bold

  # พล็อตจุด Depot
  depot_row = df_original.iloc[0]
  fig.add_trace(
      ScatterTrace(
          lat=[depot_row["Latitude"]],
          lon=[depot_row["Longitude"]],
          mode="markers+text",
          marker=dict(size=14, color="red"),
          text=["DEPOT"],
          textposition="bottom center",
          hovertext=[f"🏢 คลังสินค้า: {depot_row['name']}"],
          name="คลังสินค้า (Depot)",
      )
  )

  # พล็อตเส้นทางแต่ละคัน
  for idx, row in all_results_df.iterrows():
    sub_df = row["sub_df"]
    route = row["route"]
    truck_name = row["truck_label"]
    color = palette[idx % len(palette)]

    route_lats = [sub_df.iloc[node]["Latitude"] for node in route]
    route_lons = [sub_df.iloc[node]["Longitude"] for node in route]

    fig.add_trace(
        ScatterTrace(
            lat=route_lats,
            lon=route_lons,
            mode="lines+markers",
            name=truck_name,
            hovertext=[
                f"Node {sub_df.iloc[node]['node']}: {sub_df.iloc[node]['name']}"
                for node in route
            ],
            line=dict(width=3, color=color),
            marker=dict(size=7, color=color),
        )
    )

  center_lat = df_original["Latitude"].mean()
  center_lon = df_original["Longitude"].mean()

  layout_settings = {
      map_layer_key: dict(
          style="open-street-map",
          center=dict(lat=center_lat, lon=center_lon),
          zoom=5.5,
      ),
      "margin": {"r": 0, "t": 0, "l": 0, "b": 0},
      "height": 550,
      "showlegend": True,
  }
  fig.update_layout(**layout_settings)
  return fig


# ==========================================
# 4. ส่วน Sidebar (ใช้ Native ทั้งหมด ไม่บั๊ก)
# ==========================================
with st.sidebar:
  st.title("🚚 Fleet Setting")
  st.write("กำหนดค่ากองรถและการคำนวณ")

  uploaded_file = st.file_uploader(
      "อัปโหลดไฟล์ Excel / CSV", type=["xlsx", "csv"]
  )

  st.divider()
  st.subheader("พิกัดความจุรถ")
  col_c1, col_c2 = st.columns(2)
  cap_4w = col_c1.number_input("4 ล้อ (kg)", value=2000.0, step=100.0)
  cap_6w = col_c2.number_input("6 ล้อ (kg)", value=6000.0, step=500.0)

  st.subheader("ค่าน้ำมัน")
  fuel_price = st.number_input(
      "ดีเซล (บาท/ลิตร)", value=33.0, step=0.5
  )

  st.divider()
  routing_mode = st.radio(
      "โหมดการจัดเส้นทาง",
      ("แบ่งโซนก่อนจัดเส้นทาง (แนะนำ)", "จัดเส้นทางรวมทั้งประเทศ"),
  )

  algorithm_choice = st.selectbox(
      "อัลกอริทึม",
      ("Clarke & Wright's Savings", "Nearest Neighbor (NN)"),
  )

  run_calc = st.button("🚀 เริ่มประมวลผล", type="primary", use_container_width=True)


# ==========================================
# 5. พื้นที่ทำงานหลัก (Main Content)
# ==========================================
st.title("🚚 Vehicle Routing & Fuel Optimization Dashboard")
st.caption(
    "ระบบวิเคราะห์เส้นทางการขนส่ง จัดสรรประเภทรถ (4 ล้อ / 6 ล้อ)"
    " และคำนวณอัตราสิ้นเปลืองน้ำมัน"
)
st.divider()

if uploaded_file is not None:
  df = (
      pd.read_excel(uploaded_file)
      if uploaded_file.name.endswith(".xlsx")
      else pd.read_csv(uploaded_file)
  )
  df["Demand_Cleaned"] = df["Demand"].apply(clean_demand)
  df["Zone"] = [
      assign_zone(lat, lon, node)
      for lat, lon, node in zip(df["Latitude"], df["Longitude"], df["node"])
  ]

  if run_calc or "results_df" not in st.session_state:
    all_routes_data = []
    truck_counter = 1

    if routing_mode == "แบ่งโซนก่อนจัดเส้นทาง (แนะนำ)":
      active_zones = [z for z in df["Zone"].unique() if "Depot" not in z]
      for zone_name in active_zones:
        zone_df = (
            pd.concat([df[df["node"] == 0], df[df["Zone"] == zone_name]])
            .drop_duplicates()
            .reset_index(drop=True)
        )
        dist_matrix = get_distance_matrix(zone_df)
        demands = zone_df["Demand_Cleaned"].values

        routes = (
            solve_clarke_wright(dist_matrix, demands, cap_6w)
            if algorithm_choice == "Clarke & Wright's Savings"
            else solve_nearest_neighbor(dist_matrix, demands, cap_6w)
        )
        is_urban = "กรุงเทพ" in zone_name

        for r in routes:
          r_dist = calculate_route_distance(r, dist_matrix)
          r_load = sum([demands[i] for i in r])
          truck_type, km_per_l, fuel_l = calculate_fuel_and_truck(
              r_dist, r_load, cap_4w, cap_6w, is_urban
          )
          cost = fuel_l * fuel_price

          all_routes_data.append({
              "คันที่": f"คันที่ {truck_counter}",
              "โซน": zone_name,
              "ประเภทรถ": truck_type,
              "สภาพพื้นที่": "ในเมือง/รถติด" if is_urban else "ทางยาว/ต่างจังหวัด",
              "น้ำหนักบรรทุก (kg)": round(r_load, 2),
              "ระยะทาง (กม.)": round(r_dist, 2),
              "อัตราสิ้นเปลือง (กม./ลิตร)": km_per_l,
              "น้ำมันที่ใช้ (ลิตร)": round(fuel_l, 2),
              "ค่าน้ำมัน (บาท)": round(cost, 2),
              "route": r,
              "truck_label": (
                  f"คันที่ {truck_counter}: {truck_type} ({zone_name.replace('โซน', '')})"
              ),
              "sub_df": zone_df,
          })
          truck_counter += 1
    else:
      dist_matrix = get_distance_matrix(df)
      demands = df["Demand_Cleaned"].values
      routes = (
          solve_clarke_wright(dist_matrix, demands, cap_6w)
          if algorithm_choice == "Clarke & Wright's Savings"
          else solve_nearest_neighbor(dist_matrix, demands, cap_6w)
      )

      for idx, r in enumerate(routes):
        r_dist = calculate_route_distance(r, dist_matrix)
        r_load = sum([demands[i] for i in r])
        truck_type, km_per_l, fuel_l = calculate_fuel_and_truck(
            r_dist, r_load, cap_4w, cap_6w, False
        )
        cost = fuel_l * fuel_price

        all_routes_data.append({
            "คันที่": f"คันที่ {idx + 1}",
            "โซน": "รวมทั่วประเทศ",
            "ประเภทรถ": truck_type,
            "สภาพพื้นที่": "ทางยาว/ต่างจังหวัด",
            "น้ำหนักบรรทุก (kg)": round(r_load, 2),
            "ระยะทาง (กม.)": round(r_dist, 2),
            "อัตราสิ้นเปลือง (กม./ลิตร)": km_per_l,
            "น้ำมันที่ใช้ (ลิตร)": round(fuel_l, 2),
            "ค่าน้ำมัน (บาท)": round(cost, 2),
            "route": r,
            "truck_label": f"คันที่ {idx + 1}: {truck_type}",
            "sub_df": df,
        })

    st.session_state["results_df"] = pd.DataFrame(all_routes_data)

  results_df = st.session_state["results_df"]

  # ตัวชี้วัดสำคัญ (Native st.metric คลีน 100%)
  tot_dist = results_df["ระยะทาง (กม.)"].sum()
  tot_fuel = results_df["น้ำมันที่ใช้ (ลิตร)"].sum()
  tot_cost = results_df["ค่าน้ำมัน (บาท)"].sum()
  c_4w = (results_df["ประเภทรถ"] == "รถ 4 ล้อ").sum()
  c_6w = (results_df["ประเภทรถ"] == "รถ 6 ล้อ").sum()

  k1, k2, k3, k4 = st.columns(4)
  with k1:
    with st.container(border=True):
      st.metric(
          "📍 ระยะทางรวมทั้งหมด",
          f"{tot_dist:,.1f} กม.",
          f"เฉลี่ย {tot_dist / len(results_df):,.1f} กม./คัน",
          delta_color="off",
      )
  with k2:
    with st.container(border=True):
      st.metric(
          "⛽ น้ำมันที่ใช้โดยประมาณ",
          f"{tot_fuel:,.1f} ลิตร",
          "มาตรฐานดีเซล B7",
          delta_color="off",
      )
  with k3:
    with st.container(border=True):
      st.metric(
          "💰 ค่าน้ำมันรวมทั้งหมด",
          f"฿{tot_cost:,.0f}",
          f"ราคา ฿{fuel_price}/ลิตร",
          delta_color="off",
      )
  with k4:
    with st.container(border=True):
      st.metric(
          "🚛 จำนวนรถที่จัดสรร",
          f"{len(results_df)} คัน",
          f"4 ล้อ: {c_4w} | 6 ล้อ: {c_6w}",
          delta_color="off",
      )

  st.write("")

  # แท็บจัดการหมวดหมู่
  tab_map, tab_analytics, tab_dispatch, tab_data = st.tabs([
      "🗺️ แผนที่เส้นทาง",
      "📊 กราฟวิเคราะห์ต้นทุน",
      "🚚 แผนการปล่อยรถ (Google Maps)",
      "📋 ตารางข้อมูลสรุป",
  ])

  with tab_map:
    with st.container(border=True):
      st.subheader("🗺️ แผนที่เส้นทางการเดินรถจริง")
      st.plotly_chart(
          plot_clean_map(results_df, df),
          use_container_width=True,
          config={"displayModeBar": False},
      )

  with tab_analytics:
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
      with st.container(border=True):
        st.subheader("ค่าน้ำมันแยกตามโซน (บาท)")
        fuel_zone = (
            results_df.groupby("โซน")["ค่าน้ำมัน (บาท)"].sum().reset_index()
        )
        fig_b = px.bar(
            fuel_zone,
            x="โซน",
            y="ค่าน้ำมัน (บาท)",
            color="โซน",
            text_auto=".2s",
        )
        fig_b.update_layout(showlegend=False, height=330)
        st.plotly_chart(
            fig_b, use_container_width=True, config={"displayModeBar": False}
        )

    with c_chart2:
      with st.container(border=True):
        st.subheader("สัดส่วนการใช้รถ (4 ล้อ vs 6 ล้อ)")
        truck_pie = results_df["ประเภทรถ"].value_counts().reset_index()
        truck_pie.columns = ["ประเภทรถ", "จำนวนคัน"]
        fig_p = px.pie(
            truck_pie,
            names="ประเภทรถ",
            values="จำนวนคัน",
            hole=0.45,
            color="ประเภทรถ",
        )
        fig_p.update_layout(height=330)
        st.plotly_chart(
            fig_p, use_container_width=True, config={"displayModeBar": False}
        )

  with tab_dispatch:
    with st.container(border=True):
      st.subheader("🚚 แผนการปล่อยรถและจุดส่งแต่ละคัน")
      for idx, row in results_df.iterrows():
        sub_df = row["sub_df"]
        route = row["route"]
        route_names = [f"#{sub_df.iloc[node]['node']}" for node in route]
        gmaps_url = make_google_maps_link(sub_df, route)

        with st.expander(
            f"🚛 {row['truck_label']}  |  ระยะทาง: {row['ระยะทาง (กม.)']:,} กม.  |"
            f"  บรรทุก: {row['น้ำหนักบรรทุก (kg)']:,} kg  |  ค่าน้ำมัน:"
            f" ฿{row['ค่าน้ำมัน (บาท)']:,}"
        ):
          st.write(
              f"**โซน:** {row['โซน']} ({row['สภาพพื้นที่']}) |"
              f" **อัตราสิ้นเปลือง:** {row['อัตราสิ้นเปลือง (กม./ลิตร)']} กม./ลิตร"
          )
          st.write(f"**ลำดับจุดส่ง:** {' ➔ '.join(route_names)}")
          st.link_button(
              "📍 เปิดนำทางบน Google Maps", gmaps_url, use_container_width=False
          )

  with tab_data:
    with st.container(border=True):
      st.subheader("📋 ตารางวิเคราะห์สายการเดินรถ")
      cols_to_show = [
          "คันที่",
          "โซน",
          "ประเภทรถ",
          "สภาพพื้นที่",
          "น้ำหนักบรรทุก (kg)",
          "ระยะทาง (กม.)",
          "อัตราสิ้นเปลือง (กม./ลิตร)",
          "น้ำมันที่ใช้ (ลิตร)",
          "ค่าน้ำมัน (บาท)",
      ]
      st.dataframe(
          results_df[cols_to_show], use_container_width=True, hide_index=True
      )

else:
  st.info(
      "👈 กรุณาอัปโหลดไฟล์ Excel / CSV ที่แถบด้านซ้าย เพื่อเริ่มต้นใช้งานแดชบอร์ด"
  )