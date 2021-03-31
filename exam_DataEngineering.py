import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import geopandas as gpd
from geopandas.tools import sjoin
import rtree
import seaborn as sns
from scipy import optimize
from math import *
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("./Meteorite_Landings.csv")
data = data.dropna() 
print(data)

data["year"] = data["year"].str[6:10]
data["year"] = data["year"].astype(int)

#1 



data_year = data[data["year"] >= 1975]
data_year = data_year[data_year["year"] <=2050]

data_q1a = data[data['mass (g)'] >= 0.0]
data_q1b = data[data['mass (g)'] <= 50000.0]
 
#Histogram mass distribution (All)
plt.figure("Histogram mass distribution (All)")
ax = data_q1a["mass (g)"].plot(kind="hist", bins=500)
ax.set_yscale('log')
 
#Histogram mass distribution (Less than 50 000 grams)
plt.figure("Histogram mass distribution (Less than 50 000 grams)")
ax = data_q1b["mass (g)"].plot(kind="hist", bins=50)
ax.set_yscale('log')

#Density
#plt.figure("Density")
#axis = data_q1b["mass (g)"].plot.kde()
 
print(f"Pour la question 1a on a :\n{data_q1a.shape}\n")
print(f"Pour la question 1b on a :\n{data_q1b.shape}\n")
print(data.dtypes)
 
print(data.shape)



#2

CountOfFallByYears = data_year.groupby(['year'])['year'].count().reset_index(name="count")
print(CountOfFallByYears)
plt.figure("year(All)")
X = CountOfFallByYears.iloc[:, 0:1].values
Y = CountOfFallByYears.iloc[:, 1].values  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
plt.figure("year(All)")
plt.scatter(CountOfFallByYears['year'],CountOfFallByYears['count'])
plt.plot(X, Y_pred, color='red')


next_year = 2022
Nbcrash2022 = linear_regressor.intercept_+linear_regressor.coef_*next_year
print(Nbcrash2022)


#3
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
Oman = world.loc[world['name'] == 'Oman'] # get Oman row
boundaries = Oman['geometry'] # get Oman geometry
type(Oman)
gpd.geodataframe.GeoDataFrame
Oman.head()

tmp=[]
tmp.append([])
tmp.append([])
for coord in data["GeoLocation"]:
    partial=coord.split(",")
    x=partial[0]
    x=x[1:]
    tmp[0].append(x)
    y=partial[1]
    y=y[1:-1]
    tmp[1].append(y)


points = gpd.GeoDataFrame(data, geometry = gpd.points_from_xy(tmp[1],tmp[0])) 
points.head()

pointInPolys = sjoin(points, Oman, how='inner')
pointInPolys.head()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_aspect('equal')
Oman.plot(ax=ax, color='white', edgecolor='black')
ax.set_title('Carte avec geopandas')
pointInPolys.plot(ax=ax, color='red' ) 
#4
X=pointInPolys['geometry'].x

fig, ax1 = plt.subplots()
sns.kdeplot(data=X, ax=ax1)
ax1.set_xlim((X.min(), 55.2))
ax2 = ax1.twinx()
sns.histplot(data=X, discrete=True, ax=ax2)

Y=pointInPolys['geometry'].y

fig, ax1 = plt.subplots()
sns.kdeplot(data=Y, ax=ax1)
ax1.set_xlim((Y.min(), 20.2))
ax2 = ax1.twinx()
sns.histplot(data=Y, discrete=True, ax=ax2)


#p=fitgaussian(pointInPolys['geometry'])

DataGeo = np.transpose([X,Y])


centre_x =54.58
centre_y =19.14
A=sqrt(953*953+1358*1358)
ecart_x = (54.58-53.52)/0.64
ecart_y = (19.14-17.86)/2
print(A,ecart_x,ecart_y)

#Parameters to set

mu_x = 54.58

variance_x = 0.3



mu_y = 19.14

variance_y = 0.64



#Create grid and multivariate normal

x = np.linspace(53,56,500)

y = np.linspace(17,21,500)

X, Y = np.meshgrid(x,y)

pos = np.empty(X.shape + (2,))

pos[:, :, 0] = X; pos[:, :, 1] = Y

rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])



#Make a 3D plot

fig = plt.figure()

ax = fig.gca(projection='3d')

ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)

ax.set_xlabel('X axis')

ax.set_ylabel('Y axis')

ax.set_zlabel('Z axis')
import math 

def getIndexPoint(rv, a, b, R):
  K = 499
  Sum = 0
  CountSum = 0
  for i in range(K):
    for j in range(K):
      if (math.sqrt(((pos[:, :, 0][i][j])-a)**2+((pos[:, :, 1][i][j])-b)**2))<R:
        Sum += rv.pdf(pos[i][j])
        CountSum +=1
  return (Sum/CountSum)

a = getIndexPoint(rv, 53.9555, 18.9644, 1)
print(a)

plt.show()


    









