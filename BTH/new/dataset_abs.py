import numpy as np
import numpy.ma as ma

data = np.zeros((44,363,42)) #第一天的数据缺少；并且由于有bias_ystd变量，再删去第二天数据
#污染物观测 NO2,SO2,O3,PM25,PM10,CO(mg/m3)
p_obs = np.load('D:/project/data/pollutant/obs/p_obs_daily.npy') #(232, 182, 365, 6)
#污染物模拟 PM25,PM10,SO2,NO2,O3,CO(mg/m3)
p_sim = np.load('D:/project/data/pollutant/cmaq/p_sim_daily.npy') #(232, 182, 365, 6)
sim_var = [3, 2, 4, 0, 1, 5]
p_loc = [(129,99),(130,99),(139,100),(141,100),(135,101),(141,101),(138,102),(139,102),(141,102),(130,103),(141,103),(145,103),(146,103),(136,105),(141,105),(142,105),(127,106),(134,106),(143,106),(144,106),(127,107),(138,109),(133,111),(139,111),(138,112),(139,112),(140,112),(138,113),(140,113),(137,114),(135,115),(136,115),(141,115),(142,115),(135,116),(136,116),(137,116),(146,116),(135,117),(136,117),(131,119),(140,120),(144,125),(143,126)] #44个
m_loc = [(129,99),(129,99),(139,100),(142,100),(134,100),(142,100),(138,102),(138,102),(140,102),(130,104),(140,102),(146,103),(146,103),(137,105),(140,106),(143,106),(127,106),(133,105),(143,106),(143,106),(127,106),(139,109),(133,110),(138,112),(138,112),(140,112),(140,112),(138,112),(140,112),(136,115),(136,115),(136,115),(141,115),(141,115),(134,117),(137,117),(137,117),(146,116),(134,117),(137,117),(131,119),(140,120),(143,126),(143,126)]#p_loc对应的最邻近气象站点

#for loc in p_loc:
    #print(loc, np.sum(p_obs[loc[0],loc[1]] == -999.), np.where(p_obs[loc[0],loc[1]] == -999.))

#(141, 101) 11 (array([  0,   0,   0,   0,   0,   0, 113, 114, 115, 116, 117], dtype=int64), array([0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5], dtype=int64))
for i in range(113,118):
    p_obs[141,101,i,5] = p_obs[141,101,112,5] / p_sim[141,101,112,sim_var[5]] * p_sim[141,101,i,sim_var[5]]
#print(p_obs[141,101,113:118,5])
#delete (140, 102) 30 (array([  0,   0,   0,   0,   0,   0, 140, 141, 142, 143, 143, 143, 278,278, 278, 279, 279, 279, 292, 292, 292, 292, 292, 292, 293, 293, 293, 293, 293, 293], dtype=int64), array([0, 1, 2, 3, 4, 5, 1, 1, 1, 0, 1, 5, 0, 1, 5, 0, 1, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=int64))
#(141, 102) 12 (array([  0,   0,   0,   0,   0,   0, 220, 220, 220, 220, 220, 220], dtype=int64), array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=int64))
for i in range(6):
    p_obs[141,102,220,i] = p_obs[141,102,221,i] / p_sim[141,102,221,sim_var[i]] * p_sim[141,102,220,sim_var[i]]
#print(p_obs[141,102,220,:])
#(141, 103) 12 (array([ 0,  0,  0,  0,  0,  0, 85, 85, 85, 85, 85, 85], dtype=int64), array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=int64))
for i in range(6):
    p_obs[141,103,85,i] = p_obs[141,103,86,i] / p_sim[141,103,86,sim_var[i]] * p_sim[141,103,85,sim_var[i]]
#print(p_obs[141,103,85,:])
#(144, 106) 8 (array([  0,   0,   0,   0,   0,   0, 347, 348], dtype=int64), array([0, 1, 2, 3, 4, 5, 3, 3], dtype=int64))
for i in range(347,349):
    p_obs[144,106,i,3] = p_obs[144,106,349,3] / p_sim[144,106,349,sim_var[3]] * p_sim[144,106,i,sim_var[3]]
#print(p_obs[144,106,347:349,3])
#delete (138, 111) 49
#(139, 111) 18 (array([  0,   0,   0,   0,   0,   0, 155, 155, 155, 155, 155, 155, 171, 171, 171, 171, 171, 171], dtype=int64), array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=int64))
for i in range(6):
    p_obs[139,111,155,i] = p_obs[139,111,156,i] / p_sim[139,111,156,sim_var[i]] * p_sim[139,111,155,sim_var[i]]
    p_obs[139,111,171,i] = p_obs[139,111,172,i] / p_sim[139,111,172,sim_var[i]] * p_sim[139,111,171,sim_var[i]]
#print(p_obs[139,111,155,:])
#print(p_obs[139,111,171,:])
#(140, 112) 7 (array([  0,   0,   0,   0,   0,   0, 101], dtype=int64), array([0, 1, 2, 3, 4, 5, 4], dtype=int64))
#(138, 113) 24 (array([  0,   0,   0,   0,   0,   0, 207, 207, 207, 207, 207, 207, 208,208, 208, 208, 208, 208, 209, 209, 209, 209, 209, 209], dtype=int64), array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=int64))
for i in range(6):
    p_obs[138,113,207,i] = p_obs[138,113,206,i] / p_sim[138,113,206,sim_var[i]] * p_sim[138,113,207,sim_var[i]]
    p_obs[138,113,208,i] = p_obs[138,113,206,i] / p_sim[138,113,206,sim_var[i]] * p_sim[138,113,208,sim_var[i]]
    p_obs[138,113,209,i] = p_obs[138,113,206,i] / p_sim[138,113,206,sim_var[i]] * p_sim[138,113,209,sim_var[i]]
#print(p_obs[138,113,207,:])
#print(p_obs[138,113,208,:])
#print(p_obs[138,113,209,:])
#(140, 113) 36 (array([  0,   0,   0,   0,   0,   0,  98,  98,  98,  98,  98,  98,  99,99,  99,  99,  99,  99, 100, 100, 100, 100, 100, 100, 101, 101,101, 101, 101, 101, 147, 147, 147, 147, 147, 147], dtype=int64), array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=int64))
for i in range(6):
    p_obs[140,113,98,i] = p_obs[140,113,97,i] / p_sim[140,113,97,sim_var[i]] * p_sim[140,113,98,sim_var[i]]
    p_obs[140,113,99,i] = p_obs[140,113,97,i] / p_sim[140,113,97,sim_var[i]] * p_sim[140,113,99,sim_var[i]]
    p_obs[140,113,100,i] = p_obs[140,113,97,i] / p_sim[140,113,97,sim_var[i]] * p_sim[140,113,100,sim_var[i]]
    p_obs[140,113,101,i] = p_obs[140,113,97,i] / p_sim[140,113,97,sim_var[i]] * p_sim[140,113,101,sim_var[i]]
    p_obs[140,113,147,i] = p_obs[140,113,97,i] / p_sim[140,113,97,sim_var[i]] * p_sim[140,113,147,sim_var[i]]
#print(p_obs[140,113,98,:])
#print(p_obs[140,113,99,:])
#print(p_obs[140,113,100,:])
#print(p_obs[140,113,101,:])
#print(p_obs[140,113,147,:])
#(135, 115) 22 (array([  0,   0,   0,   0,   0,   0,  50, 179, 220, 308, 309, 312, 316,317, 323, 324, 325, 333, 341, 342, 347, 360], dtype=int64), array([0, 1, 2, 3, 4, 5, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],dtype=int64))
p_obs[135,115,220,0] = p_obs[135,115,221,0] / p_sim[135,115,221,sim_var[0]] * p_sim[135,115,220,sim_var[0]]
#print(p_obs[135,115,220,0])
#(136, 115) 13 (array([  0,   0,   0,   0,   0,   0,  50, 308, 317, 323, 325, 342, 343], dtype=int64), array([0, 1, 2, 3, 4, 5, 4, 4, 4, 4, 4, 4, 4], dtype=int64))
#(135, 116) 11 (array([  0,   0,   0,   0,   0,   0,  50, 271, 279, 323, 325], dtype=int64), array([0, 1, 2, 3, 4, 5, 4, 4, 4, 4, 4], dtype=int64))
#(136, 116) 17 (array([  0,   0,   0,   0,   0,   0,  50, 178, 279, 289, 308, 316, 323,324, 325, 342, 343], dtype=int64), array([0, 1, 2, 3, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=int64))
#(137, 116) 40 (array([  0,   0,   0,   0,   0,   0,  13,  14,  24,  44,  45,  50, 178,201, 202, 289, 293, 308, 309, 311, 313, 314, 315, 316, 317, 318, 323, 324, 325, 331, 332, 333, 341, 342, 343, 352, 353, 354, 356,363], dtype=int64), array([0, 1, 2, 3, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=int64))
#(135, 117) 30 (array([  0,   0,   0,   0,   0,   0,  14,  24,  31,  44,  45,  50, 201,273, 274, 275, 276, 277, 278, 279, 289, 309, 310, 313, 318, 323,324, 325, 342, 343], dtype=int64), array([0, 1, 2, 3, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4, 4, 4, 4, 4, 4, 4, 4], dtype=int64))
#(136, 117) 33 (array([  0,   0,   0,   0,   0,   0,  13,  15, 169, 170, 171, 175, 176, 201, 308, 309, 313, 315, 316, 317, 318, 323, 324, 331, 332, 333, 334, 340, 341, 342, 343, 353, 354], dtype=int64), array([0, 1, 2, 3, 4, 5, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=int64))
for i in range(169,172):
    p_obs[136,117,i,2] = p_obs[136,117,168,2] / p_sim[136,117,168,sim_var[2]] * p_sim[136,117,i,sim_var[2]]
    #print(p_obs[136,117,i,2])
#(146, 117) 16
#(144, 125) 7 (array([  0,   0,   0,   0,   0,   0, 185], dtype=int64), array([0, 1, 2, 3, 4, 5, 5], dtype=int64))
p_obs[144,125,185,5] = p_obs[144,125,184,5] / p_sim[144,125,184,sim_var[5]] * p_sim[144,125,185,sim_var[5]]
#print(p_obs[144,125,185,5])

# for loc in p_loc:
#     print(loc, np.sum(p_obs[loc[0],loc[1]] == -999.), np.where(p_obs[loc[0],loc[1]] == -999.))
#气象观测 PRE,RH,WSPD,WDIR,TEM
mete_obs = np.load('D:/project/data/mete/obs/cma/m_obs_daily.npy')
#for loc in m_loc:
    #print(loc, np.sum(mete_obs[loc[0],loc[1]] == -999.), np.where(mete_obs[loc[0],loc[1]] == -999.)) #都有值
#气象模拟 RH,TEM,PBLH,SOL_RAD,PRE,WSPD,WDIR
mete_sim = np.load('D:/project/data/mete/cmaq/m_sim_daily.npy')
for i in range(44):
    data[i,:,0] = (p_sim[p_loc[i][0],p_loc[i][1],2:,0] - p_obs[p_loc[i][0],p_loc[i][1],2:,3]) #PM2.5_Bias
    data[i,:,1] = (p_sim[p_loc[i][0],p_loc[i][1],2:,1] - p_obs[p_loc[i][0],p_loc[i][1],2:,4]) #PM10_Bias
    data[i,:,2] = (p_sim[p_loc[i][0],p_loc[i][1],2:,3] - p_obs[p_loc[i][0],p_loc[i][1],2:,0]) #NO2_Bias
    data[i,:,3] = (p_sim[p_loc[i][0],p_loc[i][1],2:,2] - p_obs[p_loc[i][0],p_loc[i][1],2:,1]) #SO2_Bias
    data[i,:,4] = (p_sim[p_loc[i][0],p_loc[i][1],2:,4] - p_obs[p_loc[i][0],p_loc[i][1],2:,2]) #O3_Bias
    data[i,:,5] = (p_sim[p_loc[i][0],p_loc[i][1],2:,5] - p_obs[p_loc[i][0],p_loc[i][1],2:,5]) #CO_Bias
    data[i,:,6] = p_obs[p_loc[i][0],p_loc[i][1],2:,3] #PM2.5_Obs
    data[i,:,7] = p_obs[p_loc[i][0],p_loc[i][1],2:,4] #PM10_Obs
    data[i,:,8] = p_obs[p_loc[i][0],p_loc[i][1],2:,0] #NO2_Obs
    data[i,:,9] = p_obs[p_loc[i][0],p_loc[i][1],2:,1] #SO2_Obs
    data[i,:,10] = p_obs[p_loc[i][0],p_loc[i][1],2:,2] #O3_Obs
    data[i,:,11] = p_obs[p_loc[i][0],p_loc[i][1],2:,5] #CO_Obs
    data[i,:,12] = p_sim[p_loc[i][0],p_loc[i][1],2:,0] #PM2.5_Sim
    data[i,:,13] = p_sim[p_loc[i][0],p_loc[i][1],2:,1] #PM10_Sim
    data[i,:,14] = p_sim[p_loc[i][0],p_loc[i][1],2:,3] #NO2_Sim
    data[i,:,15] = p_sim[p_loc[i][0],p_loc[i][1],2:,2] #SO2_Sim
    data[i,:,16] = p_sim[p_loc[i][0],p_loc[i][1],2:,4] #O3_Sim
    data[i,:,17] = p_sim[p_loc[i][0],p_loc[i][1],2:,5] #CO_Sim
    data[i,:,18] = (mete_sim[m_loc[i][0],m_loc[i][1],2:,0] - mete_obs[m_loc[i][0],m_loc[i][1],2:,1]) #RH_Bias
    data[i,:,19] = (mete_sim[m_loc[i][0],m_loc[i][1],2:,1] - mete_obs[m_loc[i][0],m_loc[i][1],2:,4]) #TEM_Bias
    data[i,:,20] = (mete_sim[m_loc[i][0],m_loc[i][1],2:,5] - mete_obs[m_loc[i][0],m_loc[i][1],2:,2]) #WSPD_Bias
    for j in range(2,365):  #WDIR_Bias
        WDIR_Bias = np.abs(mete_sim[m_loc[i][0],m_loc[i][1],j,6] - mete_obs[m_loc[i][0],m_loc[i][1],j,3])
        if WDIR_Bias > 180.:  
            data[i,:,21] = 360. - WDIR_Bias  
        else:
            data[i,:,21] = WDIR_Bias

    data[i,:,22] = (mete_sim[m_loc[i][0],m_loc[i][1],2:,4] - mete_obs[m_loc[i][0],m_loc[i][1],2:,0]) #PRE_Bias
    data[i,:,23] = mete_obs[m_loc[i][0],m_loc[i][1],2:,1] #RH_Obs
    data[i,:,24] = mete_obs[m_loc[i][0],m_loc[i][1],2:,4] #TEM_Obs
    data[i,:,25] = mete_obs[m_loc[i][0],m_loc[i][1],2:,2] #WSPD_Obs
    data[i,:,26] = mete_obs[m_loc[i][0],m_loc[i][1],2:,3] #WDIR_Obs
    data[i,:,27] = mete_obs[m_loc[i][0],m_loc[i][1],2:,0] #PRE_Obs
    data[i,:,28] = mete_sim[m_loc[i][0],m_loc[i][1],2:,2] #PBLH_Sim
    data[i,:,29] = mete_sim[m_loc[i][0],m_loc[i][1],2:,3] #SOLRAD_Sim
    data[i,:,30] = mete_sim[m_loc[i][0],m_loc[i][1],2:,0] #RH_Sim
    data[i,:,31] = mete_sim[m_loc[i][0],m_loc[i][1],2:,1] #TEM_Sim
    data[i,:,32] = mete_sim[m_loc[i][0],m_loc[i][1],2:,5] #WSPD_Sim
    data[i,:,33] = mete_sim[m_loc[i][0],m_loc[i][1],2:,6] #WDIR_Sim
    data[i,:,34] = mete_sim[m_loc[i][0],m_loc[i][1],2:,4] #PRE_Sim

    #风向风速矢量分解，新加6个变量
    data[i,:,35] = mete_obs[m_loc[i][0],m_loc[i][1],2:,2] * np.cos(mete_obs[m_loc[i][0],m_loc[i][1],2:,3] * np.pi / 180) #WIN_N_Obs
    data[i,:,36] = mete_sim[m_loc[i][0],m_loc[i][1],2:,5] * np.cos(mete_sim[m_loc[i][0],m_loc[i][1],2:,6] * np.pi / 180) #WIN_N_Sim
    data[i,:,37] = mete_obs[m_loc[i][0],m_loc[i][1],2:,2] * np.sin(mete_obs[m_loc[i][0],m_loc[i][1],2:,3] * np.pi / 180) #WIN_E_Obs
    data[i,:,38] = mete_sim[m_loc[i][0],m_loc[i][1],2:,5] * np.sin(mete_sim[m_loc[i][0],m_loc[i][1],2:,6] * np.pi / 180) #WIN_E_Sim
    data[i,:,39] = data[i,:,36] - data[i,:,35] #WIN_N_Bias
    data[i,:,40] = data[i,:,38] - data[i,:,37] #WIN_E_Bias
    #前一天的PM25bias
    data[i,0,41] = p_sim[p_loc[i][0],p_loc[i][1],1,0] - p_obs[p_loc[i][0],p_loc[i][1],1,3]
    data[i,1:,41] = data[i,0:362,0]

#修补WSPD_Obs data
data[13,22,25] = 1.7
data[16,322,25] = 3.0
data[20,322,25] = 4.0
data[13,323,25] = 4
data[16,323,25] = 4
data[20,323,25] = 5
data[14,324,25] = 4
#修补WIN_N_Obs data
data[13,22,35] = data[13,22,25] * np.cos(mete_obs[m_loc[13][0],m_loc[13][1],24,3] * np.pi / 180)
data[16,322,35] = data[16,322,35] * np.cos(mete_obs[m_loc[16][0],m_loc[16][1],324,3] * np.pi / 180)
data[20,322,35] = data[20,322,35] * np.cos(mete_obs[m_loc[20][0],m_loc[20][1],324,3] * np.pi / 180)
data[13,323,35] = data[13,323,35] * np.cos(mete_obs[m_loc[13][0],m_loc[13][1],325,3] * np.pi / 180)
data[16,323,35] = data[16,323,35] * np.cos(mete_obs[m_loc[16][0],m_loc[16][1],325,3] * np.pi / 180)
data[20,323,35] = data[20,323,35] * np.cos(mete_obs[m_loc[20][0],m_loc[20][1],325,3] * np.pi / 180)
data[14,324,35] = data[14,324,35] * np.cos(mete_obs[m_loc[14][0],m_loc[14][1],326,3] * np.pi / 180)
#修补WIN_E_Obs data
data[13,22,37] = data[13,22,25] * np.sin(mete_obs[m_loc[13][0],m_loc[13][1],24,3] * np.pi / 180)
data[16,322,37] = data[16,322,35] * np.sin(mete_obs[m_loc[16][0],m_loc[16][1],324,3] * np.pi / 180)
data[20,322,37] = data[20,322,35] * np.sin(mete_obs[m_loc[20][0],m_loc[20][1],324,3] * np.pi / 180)
data[13,323,37] = data[13,323,35] * np.sin(mete_obs[m_loc[13][0],m_loc[13][1],325,3] * np.pi / 180)
data[16,323,37] = data[16,323,35] * np.sin(mete_obs[m_loc[16][0],m_loc[16][1],325,3] * np.pi / 180)
data[20,323,37] = data[20,323,35] * np.sin(mete_obs[m_loc[20][0],m_loc[20][1],325,3] * np.pi / 180)
data[14,324,37] = data[14,324,35] * np.sin(mete_obs[m_loc[14][0],m_loc[14][1],326,3] * np.pi / 180)
#修补WIN_N_Bias data
data[13,22,39] = data[13,22,36] - data[13,22,35]
data[16,322,39] = data[16,322,36] - data[16,322,35]
data[20,322,39] = data[20,322,36] - data[20,322,35]
data[13,323,39] = data[13,323,36] - data[13,323,35]
data[16,323,39] = data[16,323,36] - data[16,323,35]
data[20,323,39] = data[20,323,36] - data[20,323,35]
data[14,324,39] = data[14,324,36] - data[14,324,35]
#修补WIN_E_Bias data
data[13,22,40] = data[13,22,38] - data[13,22,37]
data[16,322,40] = data[16,322,38] - data[16,322,37]
data[20,322,40] = data[20,322,38] - data[20,322,37]
data[13,323,40] = data[13,323,38] - data[13,323,37]
data[16,323,40] = data[16,323,38] - data[16,323,37]
data[20,323,40] = data[20,323,38] - data[20,323,37]
data[14,324,40] = data[14,324,38] - data[14,324,37]

print(data.shape) #(44, 363, 42)
data = np.around(data, 2) #都保留2位小数
np.save("D:/project/data/BTH/new/dataset_abs.npy", data)
