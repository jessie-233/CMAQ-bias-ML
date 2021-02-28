import numpy as np
import numpy.ma as ma


#查看网格（136,115）全年数据
file1 = np.load('D:/project/data/mete/obs/cma/m_obs_daily.npy') #模拟数据
print(file1.shape) #(232, 182, 365, 6) 
#PRE 降水cm(24h sum),RH相对湿度%,WINSPD风速m/s,WINDIR风向 deg,TEM温度℃
mete_name = ['PRE','RH','WINSPD','WINDIR','TEM']
for i in range(5):
    print(mete_name[i])
    print(np.around(file1[136,115,:,i],decimals=2))


print(file1[137,115,:,0]) #全是-999.
print(file1[137,115,:,1]) #全是-999.



#全年平均，看哪个网格没数据
m_obs_daily = ma.masked_array(file1, file1 == -999.)
m_obs_yearly = np.mean(m_obs_daily, axis=2)
print(m_obs_yearly.shape) #(232, 182, 5)
RH_obs_yearly = m_obs_yearly[:,:,1].transpose(1,0) #行号，列号
np.savetxt('D:/project/data/mete/obs/cma/RH_obs_yearly.csv', RH_obs_yearly, delimiter = ',')