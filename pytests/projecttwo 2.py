import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import csv

# 读取CSV文件
url = "https://github.com/ehmatthes/pcc_3e/tree/main/chapter_16/mapping_global_datasets/eq_data/world_fires_1_day.csv"  # Replace with actual URL
date = pd.read_csv(url)
# 提取纬度、经度和亮度数据
latitudes = date["Latitude"].tolist()
longitudes = date["Longitude"].tolist()
brightnesses = date["Brightness"].tolist()
# 使用Plotly创建世界地图
fig = px.scatter_geo(lat=latitudes, lon=longitudes, hover_name=brightnesses, color=brightnesses,
                     projection="natural earth", title="Global Fire Distribution",
                     labels={"Brightness": "Fire Intensity"})

# 显示地图
fig.show()
