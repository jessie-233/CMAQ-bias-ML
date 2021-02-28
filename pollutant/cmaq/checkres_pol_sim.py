import numpy as np
import numpy.ma as ma

#查看网格（136,115）全年数据
file1 = np.load('D:/project/data/pollutant/cmaq/p_sim_daily.npy') #模拟数据
print(file1.shape) #(232, 182, 365, 6) 
#'PM25_TOT'ug/m3,'PM10'ug/m3,'SO2'ug/m3,'NO2'ug/m3,'O3'ug/m3,'CO'mg/m3
pol_name = ['PM25_TOT','PM10','SO2','NO2','O3','CO']
#for i in range(6):
    #print(pol_name[i])
    #print(np.around(file1[136,115,:,i],decimals=2))

file2 = np.load('D:/project/data/pollutant/obs/p_obs_daily.npy') #观测数据
#print(file2.shape) #(232, 182, 365, 6) 
pol_name2 = ['NO2-obs','SO2-obs','O3-obs','PM25-obs','PM10-obs','CO-obs']
#for i in range(6):
    #print(pol_name2[i])
    #print(np.around(file2[136,115,:,i],decimals=2))



#全年平均，看哪个网格没数据
pm25_obs_daily = ma.masked_array(file2[:,:,:,3], file2[:,:,:,3] == -999.)
pm25_obs_yearly = np.mean(pm25_obs_daily, axis=2).transpose(1,0)
print(pm25_obs_yearly.shape) #(182, 232)
#print(p_obs_yearly.shape) #(232, 182, 6)
#print(p_obs_yearly[136,115,0]) #52.29175900419101
#NO2_obs_yearly = p_obs_yearly[:,:,0].transpose(1,0)
#print(NO2_obs_yearly.shape) #(182, 232)
#print(NO2_obs_yearly[115,136]) #52.29175900419101
#np.savetxt('D:/project/data/pollutant/obs/NO2_obs_yearly.csv', NO2_obs_yearly, delimiter = ',')
pm25_sim_daily = ma.masked_array(file1[:,:,:,0], file1[:,:,:,0] == -999.)
pm25_sim_yearly = np.mean(pm25_sim_daily, axis=2).transpose(1,0)
print(pm25_sim_yearly.shape) #(182, 232)
np.savetxt('D:/project/data/pollutant/cmaq/pm25_sim_yearly.csv', pm25_sim_yearly, delimiter = ',')
np.savetxt('D:/project/data/pollutant/obs/pm25_obs_yearly.csv', pm25_obs_yearly, delimiter = ',')









