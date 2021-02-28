import numpy as np
import os
import numpy.ma as ma

file = np.load('D:/project/data/pollutant/obs/p_obs_daily.npy')
print(file.shape) #(232, 182, 365, 6) NO2,SO2,O3,PM25,PM10,CO(mg/m3)
pollutant_daily = np.around(file,decimals = 2)

#全年平均，看哪个网格没数据
p_obs_daily = ma.masked_array(file, file == -999.)
p_obs_yearly = np.mean(p_obs_daily, axis=2)
print(p_obs_yearly.shape) #(232, 182, 6)
PM25_obs_yearly = p_obs_yearly[:,:,3].transpose(1,0) #行号，列号
np.savetxt('D:/project/data/pollutant/obs/PM25_obs_yearly.csv', PM25_obs_yearly, delimiter = ',')