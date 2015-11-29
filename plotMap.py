import pandas as pd

df = pd.read_csv('data_sample_1mb.csv')
lats, lons = [], []

lats1 = df[df.Outcome == 'w']['Latitude'].tolist()
lons1 = df[df.Outcome == 'w']['Longitude'].tolist()

lats2 = df[df.Outcome == '0']['Latitude'].tolist()
lons2 = df[df.Outcome == '0']['Longitude'].tolist()
# --- Build Map ---
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
 
eq_map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0,
              lat_0=0, lon_0=-130)
eq_map.drawcoastlines()
eq_map.drawcountries()
eq_map.fillcontinents(color = 'gray')
eq_map.drawmapboundary()
eq_map.drawmeridians(np.arange(0, 360, 30))
eq_map.drawparallels(np.arange(-90, 90, 30))
 
x,y = eq_map(lons1, lats1)
eq_map.plot(x, y, 'ro', markersize=6, color='red')

x1,y1 = eq_map(lons2, lats2)
eq_map.plot(x1, y1, 'ro', markersize=6, color='blue')

plt.show()