import numpy as np
import numpy.ma as ma
import copy
np.set_printoptions(suppress=True) #抑制使用对小数的科学记数法
np.set_printoptions(precision=2) #设置输出的精度
#目前PM10 Obs数据仍有问题

data = np.zeros((44,363,43)) #第一天的数据缺少；并且由于有bias_ystd变量，再删去第二天数据
p_loc = [(129,99),(130,99),(139,100),(141,100),(135,101),(141,101),(138,102),(139,102),(141,102),(130,103),(141,103),(145,103),(146,103),(136,105),(141,105),(142,105),(127,106),(134,106),(143,106),(144,106),(127,107),(138,109),(133,111),(139,111),(138,112),(139,112),(140,112),(138,113),(140,113),(137,114),(135,115),(136,115),(141,115),(142,115),(135,116),(136,116),(137,116),(146,116),(135,117),(136,117),(131,119),(140,120),(144,125),(143,126)] #44个
m_loc = [(129,99),(129,99),(139,100),(142,100),(134,100),(142,100),(138,102),(138,102),(140,102),(130,104),(140,102),(146,103),(146,103),(137,105),(140,106),(143,106),(127,106),(133,105),(143,106),(143,106),(127,106),(139,109),(133,110),(138,112),(138,112),(140,112),(140,112),(138,112),(140,112),(136,115),(136,115),(136,115),(141,115),(141,115),(134,117),(137,117),(137,117),(146,116),(134,117),(137,117),(131,119),(140,120),(143,126),(143,126)]#p_loc对应的最邻近气象站点


#污染物观测 NO2,SO2,O3,PM25,PM10,CO(mg/m3)
p_obs = np.load('D:/project/data/pollutant/obs/p_obs_daily.npy') #(232, 182, 365, 6)
# p_obs = p_obs[:,:,:,:4]


#污染物模拟 PM25,PM10,SO2,NO2,O3,CO(mg/m3)
p_sim = np.load('D:/project/data/pollutant/cmaq/p_sim_daily.npy') #(232, 182, 365, 6)


# for loc in p_loc:
#     print(loc, np.sum(p_obs[loc[0],loc[1]] == -999.), np.where(p_obs[loc[0],loc[1]] == -999.))

def Fix(col,row,day,item):
    global p_obs
    day_copy1 = copy.deepcopy(day)
    day_copy2 = copy.deepcopy(day)
    min_day = day
    max_day = day
    while p_obs[col,row,day_copy1,item] == -999.0:
        day_copy1 -= 1
        min_day = day_copy1
    while p_obs[col,row,day_copy2,item] == -999.0:
        day_copy2 += 1
        max_day = day_copy2
    p_obs[col,row,day,item] = (p_obs[col,row,min_day,item] + p_obs[col,row,max_day,item]) / 2
    print(p_obs[col,row,day,item])

# (129, 99) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (130, 99) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (139, 100) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (141, 100) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (135, 101) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (141, 101) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (138, 102) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (139, 102) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (141, 102) 8 (array([  0,   0,   0,   0, 220, 220, 220, 220], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
for item in range(4):
    Fix(141,102,220,item)

# (130, 103) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (141, 103) 8 (array([ 0,  0,  0,  0, 85, 85, 85, 85], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
for item in range(4):
    Fix(141,103,85,item)

# (145, 103) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (146, 103) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (136, 105) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (141, 105) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (142, 105) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (127, 106) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (134, 106) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (143, 106) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (144, 106) 6 (array([  0,   0,   0,   0, 347, 348], dtype=int64), array([0, 1, 2, 3, 3, 3], dtype=int64))
Fix(144,106,347,3)
Fix(144,106,348,3)

# (127, 107) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (138, 109) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (133, 111) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (139, 111) 12 (array([  0,   0,   0,   0, 155, 155, 155, 155, 171, 171, 171, 171],dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
for item in range(4):
    Fix(139,111,155,item)
    Fix(139,111,171,item)


# (138, 112) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (139, 112) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (140, 112) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (138, 113) 16 (array([  0,   0,   0,   0, 207, 207, 207, 207, 208, 208, 208, 208, 209,209, 209, 209], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
for item in range(4):
    Fix(138,113,207,item)
    Fix(138,113,208,item)
    Fix(138,113,209,item)


# (140, 113) 24 (array([  0,   0,   0,   0,  98,  98,  98,  98,  99,  99,  99,  99, 100,100, 100, 100, 101, 101, 101, 101, 147, 147, 147, 147], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,2, 3], dtype=int64))
for item in range(4):
    Fix(140,113,98,item)
    Fix(140,113,99,item)
    Fix(140,113,100,item)
    Fix(140,113,101,item)
    Fix(140,113,147,item)

# (137, 114) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (135, 115) 5 (array([  0,   0,   0,   0, 220], dtype=int64), array([0, 1, 2, 3, 0], dtype=int64))
Fix(135,115,220,0)

# (136, 115) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (141, 115) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (142, 115) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (135, 116) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (136, 116) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (137, 116) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (146, 116) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (135, 117) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (136, 117) 7 (array([  0,   0,   0,   0, 169, 170, 171], dtype=int64), array([0, 1, 2, 3, 2, 2, 2], dtype=int64))
Fix(136,117,169,2)
Fix(136,117,170,2)
Fix(136,117,171,2)


# (131, 119) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (140, 120) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (144, 125) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (143, 126) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))

for loc in p_loc:
    print(loc, np.sum(p_obs[loc[0],loc[1]] == -999.), np.where(p_obs[loc[0],loc[1]] == -999.))

#气象观测 PRE,RH,WSPD,WDIR,TEM
mete_obs = np.load('D:/project/data/mete/obs/cma/m_obs_daily.npy')

#check on PRE(none)
for loc in m_loc:
    a = np.sum(mete_obs[loc[0],loc[1],:,0] > 50)
    if a != 0:
        print(loc, a, np.where(mete_obs[loc[0],loc[1],:,0] > 50)) #异常值为327.66


#check on RH(none)
for loc in m_loc:
    a = np.sum(mete_obs[loc[0],loc[1],:,1] > 100)
    if a != 0:
        print(loc, a, np.where(mete_obs[loc[0],loc[1],:,1] > 100)) #无异常值

#check on WSPD
for loc in m_loc:
    a = np.sum(mete_obs[loc[0],loc[1],:,2] == 3276.6)
    if a != 0:
        print(loc, a, np.where(mete_obs[loc[0],loc[1],:,2] == 3276.6)) #异常值为3276.6

# (137, 105) 3 (array([ 24, 325, 326], dtype=int64),)
# (127, 106) 2 (array([324, 325], dtype=int64),)
# (127, 106) 2 (array([324, 325], dtype=int64),)
def Fix_WSPD(col,row,day):
    global mete_obs
    day_copy1 = copy.deepcopy(day)
    day_copy2 = copy.deepcopy(day)
    min_day = day
    max_day = day
    while mete_obs[col,row,day_copy1,2] == 3276.6:
        day_copy1 -= 1
        min_day = day_copy1
    while mete_obs[col,row,day_copy2,2] == 3276.6:
        day_copy2 += 1
        max_day = day_copy2
    mete_obs[col,row,day,2] = (mete_obs[col,row,min_day,2] + mete_obs[col,row,max_day,2]) / 2
    print(mete_obs[col,row,day,2])

Fix_WSPD(137,105,24)
Fix_WSPD(137,105,325)
Fix_WSPD(137,105,326)
Fix_WSPD(127,106,324)
Fix_WSPD(127,106,325)

#check on WDIR(none)
for loc in m_loc:
    a = np.sum(mete_obs[loc[0],loc[1],:,3] > 360)
    if a != 0:
        print(loc, a, np.where(mete_obs[loc[0],loc[1],:,3] > 360)) #异常值为737212.5

#check on TEM(none)
for loc in m_loc:
    a = np.sum(mete_obs[loc[0],loc[1],:,4] < -20) #or大于40
    if a != 0:
        print(loc, a, np.where(mete_obs[loc[0],loc[1],:,4] < -20)) #无异常值

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
            data[i,j-2,21] = 360. - WDIR_Bias  
        else:
            data[i,j-2,21] = WDIR_Bias

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

    # #风向风速矢量分解，新加6个变量
    # data[i,:,35] = mete_obs[m_loc[i][0],m_loc[i][1],2:,2] * np.cos(mete_obs[m_loc[i][0],m_loc[i][1],2:,3] * np.pi / 180) #WIN_N_Obs
    # data[i,:,36] = mete_sim[m_loc[i][0],m_loc[i][1],2:,5] * np.cos(mete_sim[m_loc[i][0],m_loc[i][1],2:,6] * np.pi / 180) #WIN_N_Sim
    # data[i,:,37] = mete_obs[m_loc[i][0],m_loc[i][1],2:,2] * np.sin(mete_obs[m_loc[i][0],m_loc[i][1],2:,3] * np.pi / 180) #WIN_E_Obs
    # data[i,:,38] = mete_sim[m_loc[i][0],m_loc[i][1],2:,5] * np.sin(mete_sim[m_loc[i][0],m_loc[i][1],2:,6] * np.pi / 180) #WIN_E_Sim
    # data[i,:,39] = data[i,:,36] - data[i,:,35] #WIN_N_Bias
    # data[i,:,40] = data[i,:,38] - data[i,:,37] #WIN_E_Bias

    #前一天的PM25bias:PM2.5_Bias_ystd
    data[i,0,35] = p_sim[p_loc[i][0],p_loc[i][1],1,0] - p_obs[p_loc[i][0],p_loc[i][1],1,3]
    data[i,1:,35] = data[i,0:362,0]

    #前一天的NO2bias:NO2_Bias_ystd
    data[i,0,36] = p_sim[p_loc[i][0],p_loc[i][1],1,3] - p_obs[p_loc[i][0],p_loc[i][1],1,0]
    data[i,1:,36] = data[i,0:362,2]

    #前一天的RHbias:RH_Bias_ystd
    data[i,0,37] = mete_sim[m_loc[i][0],m_loc[i][1],1,0] - mete_obs[m_loc[i][0],m_loc[i][1],1,1]
    data[i,1:,37] = data[i,0:362,18]

    #前一天的O3bias:O3_Bias_ystd
    data[i,0,38] = p_sim[p_loc[i][0],p_loc[i][1],1,4] - p_obs[p_loc[i][0],p_loc[i][1],1,2]
    data[i,1:,38] = data[i,0:362,4]

    #前一天的SO2bias:SO2_Bias_ystd
    data[i,0,39] = p_sim[p_loc[i][0],p_loc[i][1],1,2] - p_obs[p_loc[i][0],p_loc[i][1],1,1]
    data[i,1:,39] = data[i,0:362,3]

    #前一天的WSPDbias:WSPD_Bias_ystd
    data[i,0,40] = mete_sim[m_loc[i][0],m_loc[i][1],1,5] - mete_obs[m_loc[i][0],m_loc[i][1],1,2]
    data[i,1:,40] = data[i,0:362,20]

    #前一天的NO2Obs:NO2_Obs_ystd
    data[i,0,41] = p_obs[p_loc[i][0],p_loc[i][1],1,0]
    data[i,1:,41] = data[i,0:362,8]

    #前一天的O3Obs:O3_Obs_ystd
    data[i,0,42] = p_obs[p_loc[i][0],p_loc[i][1],1,2]
    data[i,1:,42] = data[i,0:362,10]

print(data.shape) #(44, 363, 43)
data = np.around(data, 2) #都保留2位小数
np.save("D:/project/data/BTH/DOMAIN_TRANS/BTH/dataset_BTH.npy", data)
