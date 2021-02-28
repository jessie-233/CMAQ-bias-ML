import netCDF4 as nc
import numpy as np

f = nc.Dataset("D:/project/data/mete/cmaq/GRIDCRO2D_cn27_2017001","r")
lat2d = f.variables["LAT"][0,0,:,:] #( TSTEP, LAY, ROW182, COL232 )
lon2d = f.variables["LON"][0,0,:,:]
#print(lat2d.shape) #182,232
#print(lon2d.shape) #182,232

#print(lat2d[:,0]) #8.9~50.1
#print(lat2d[0,:]) #wtf

#print(lon2d[:,0]) #83~66
#print(lon2d[0,:]) #83~136.68

print(lat2d[115,136]) #39.828568
print(lat2d[115,135]) #39.843025
