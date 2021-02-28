import numpy as np
import numpy.ma as ma
'''
不含各种sim，误差是相对的
'''

data = np.zeros((365,25))

#污染物观测 NO2,SO2,O3,PM25,PM10,CO(mg/m3)
p_obs = np.load('D:/project/data/pollutant/obs/p_obs_daily.npy')
p_obs = p_obs[132:140,113:121,:,:] #(8, 8, 365, 6)
p_obs = ma.masked_array(p_obs, p_obs == -999.) 
p_obs = np.mean(p_obs,axis=(0,1)) #(365, 6)
p_obs[0,0]= 49
p_obs[0,1]= 23
p_obs[0,2]= 29
p_obs[0,3]= 41
p_obs[0,4]= 66
p_obs[0,5]= 1.09


#污染物模拟 PM25,PM10,SO2,NO2,O3,CO(mg/m3)
p_sim = np.load('D:/project/data/pollutant/cmaq/p_sim_daily.npy')
p_sim = p_sim[132:140,113:121,:,:] #(8, 8, 365, 6)
p_sim = ma.masked_array(p_sim, p_sim == -999.) 
p_sim = np.mean(p_sim,axis=(0,1)) #(365, 6)

 
#气象观测 PRE,RH,WSPD,WDIR,TEM
mete_obs = np.load('D:/project/data/mete/obs/cma/m_obs_daily.npy')
mete_obs = mete_obs[132:140,113:121,:,:] #(8, 8, 365, 6)
mete_obs = ma.masked_array(mete_obs, mete_obs == -999.) 
mete_obs = np.mean(mete_obs,axis=(0,1)) #(365, 6)
mete_obs[324,2] = 1.31 #注意：11/21日的数据有问题，改为1.31

#气象模拟 RH,TEM,PBLH,SOL_RAD,PRE,WSPD,WDIR
mete_sim = np.load('D:/project/data/mete/cmaq/m_sim_daily.npy')
mete_sim = mete_sim[132:140,113:121,:,:] #(8, 8, 365, 6)
mete_sim = ma.masked_array(mete_sim, mete_sim == -999.) 
mete_sim = np.mean(mete_sim,axis=(0,1)) #(365, 6)


data[:,0] = (p_sim[:,0] - p_obs[:,3]) / p_obs[:,3] #PM2.5_Bias
data[:,1] = (p_sim[:,1] - p_obs[:,4]) / p_obs[:,4] #PM10_Bias
data[:,2] = (p_sim[:,3] - p_obs[:,0]) / p_obs[:,0] #NO2_Bias
data[:,3] = (p_sim[:,2] - p_obs[:,1]) / p_obs[:,1] #SO2_Bias
data[:,4] = (p_sim[:,4] - p_obs[:,2]) / p_obs[:,2] #O3_Bias
data[:,5] = (p_sim[:,5] - p_obs[:,5]) / p_obs[:,5] #CO_Bias
data[:,6] = p_obs[:,3] #PM2.5_Obs
data[:,7] = p_obs[:,4] #PM10_Obs
data[:,8] = p_obs[:,0] #NO2_Obs
data[:,9] = p_obs[:,1] #SO2_Obs
data[:,10] = p_obs[:,2] #O3_Obs
data[:,11] = p_obs[:,5] #CO_Obs
data[:,12] = p_sim[:,0] #PM2.5_Sim
data[:,13] = (mete_sim[:,0] - mete_obs[:,1]) / mete_obs[:,1] #RH_Bias
data[:,14] = (mete_sim[:,1] - mete_obs[:,4]) / mete_obs[:,4] #TEM_Bias
data[:,15] = (mete_sim[:,5] - mete_obs[:,2]) / mete_obs[:,2] #WSPD_Bias

for i in range(365):  #WDIR_Bias
    WDIR_Bias = np.abs(mete_sim[i,6] - mete_obs[i,3])
    if WDIR_Bias > 180.:  
        data[i,16] = 360. - WDIR_Bias  
    else:
        data[i,16] = WDIR_Bias

data[:,17] = (mete_sim[:,4] - mete_obs[:,0]) / mete_obs[:,0] #PRE_Bias
data[:,18] = mete_obs[:,1] #RH_Obs
data[:,19] = mete_obs[:,4] #TEM_Obs
data[:,20] = mete_obs[:,2] #WSPD_Obs
data[:,21] = mete_obs[:,3] #WDIR_Obs
data[:,22] = mete_obs[:,0] #PRE_Obs
data[:,23] = mete_sim[:,2] #PBLH_Sim
data[:,24] = mete_sim[:,3] #SOLRAD_Sim
data = np.around(data, 2) #都保留2位小数
#np.savetxt("D:/project/data/beijing/dataset.csv", data, delimiter=',')
np.save("D:/project/data/beijing/dataset.npy", data)