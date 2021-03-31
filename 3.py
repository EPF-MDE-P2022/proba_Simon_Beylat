import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import geopandas as gpd
from geopandas.tools import sjoin
import rtree
from scipy.stats import gaussian_kde

data = pd.read_excel('MeteoriteDataDL.xlsx',index_col=0)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
Oman = world.loc[world['name'] == 'Oman'] # get Oman row
boundaries = Oman['geometry'] # get Oman geometry
type(Oman)
gpd.geodataframe.GeoDataFrame
Oman.head()



points = gpd.GeoDataFrame(data, geometry = gpd.points_from_xy(data["Longitude"],data["Latitude"])) #wrapping des points gps des météorites
points.head()


pointInPolys = sjoin(points, Oman, how='inner')
pointInPolys.head()


xy = np.vstack([pointInPolys['geometry'].x,pointInPolys['geometry'].y])
z = gaussian_kde(xy)(xy)

idx = z.argsort()
x = pointInPolys['geometry'].x[idx]
y = pointInPolys['geometry'].y[idx]
z = z[idx]
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_aspect('equal')
Oman.plot(ax=ax, color='white', edgecolor='black')
ax.set_title('Carte avec geopandas');
ax.scatter(x, y, c=z, s=10)


plt.show()