import pandas as pd
import numpy as np
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy import interpolate

#1) Data Collection

veri = np.loadtxt(r'C:\Users\xboxm\vs_workspace\github_projects\gravity_method_bougier_correction\private_dataset.dat', unpack = True)


#2) Data Preparation

headers=["longitude","latitude","raw_data","elevation","top_cor"]
db=pd.DataFrame()
for i in range(len(headers)):
    db[headers[i]]=veri[i,:]


#3 & 4) DATA PROCESSING AND VISUALIZATION

gridNum = 2*len(db.latitude)
# Min max coordinates
xmin,xmax = min(db["latitude"]),max(db["latitude"])
ymin,ymax = min(db["longitude"]),max(db["longitude"])
z=np.array([])
xgrid = np.linspace(xmin, xmax, gridNum)
ygrid = np.linspace(ymin, ymax, gridNum)
xgrid, ygrid = np.meshgrid(xgrid, ygrid)
# Interpolation
ham_gravite = griddata((db["latitude"],db["longitude"]),db["raw_data"],(xgrid,ygrid), method = 'linear')


def lat_cor(phi,latitude):
    ed=0.8122*math.sin(math.radians(2*phi))
    result=np.ones(len(latitude))
    for n in range(len(latitude)):
        result[n]=(phi-latitude[n])*111*ed
    return result

def elev_cor(p0,ro,elevation):
    y=(0.3086-0.04185*ro)
    result=np.ones(len(elevation))
    for n in range(len(elevation)):
        result[n]=(elevation[n]-p0)*y
    return result

enl_duz=lat_cor(51,db.latitude)
yuk_ind=elev_cor(1000,2,db.elevation)
bouguer=[]

grs64=978031.89*(1+0.00530237*pow(math.sin(math.radians(51)),2)-0.00000585*pow(math.sin(math.radians(2*51)),2))

for n in range(len(db.raw_data)):
    bouguer.append(db.raw_data[n]+enl_duz[n]+yuk_ind[n]+db.top_cor[n]-grs64)

bouguer1 = griddata((db["latitude"],db["longitude"]),bouguer,(xgrid,ygrid), method = 'linear')


plt.figure(figsize=(13,7))
plt.subplot(1,2,1)
plt.contour(xgrid, ygrid, ham_gravite, 15, linewidths = 0.5, colors = 'k')
plt.pcolormesh(xgrid, ygrid, ham_gravite, cmap = plt.get_cmap('rainbow'))

plt.colorbar()
plt.scatter(db.latitude, db.longitude, marker = 'o', c = 'b', s = 5, zorder = 10)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.title("Raw Gravity Data (mGal)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.subplot(1,2,2)
plt.contour(xgrid, ygrid, bouguer1, 15, linewidths = 0.5, colors = 'k')
plt.pcolormesh(xgrid, ygrid, bouguer1, cmap = plt.get_cmap('rainbow'))

plt.colorbar()
plt.scatter(db.latitude, db.longitude, marker = 'o', c = 'b', s = 5, zorder = 10)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.title("Bouguer (mGal)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()