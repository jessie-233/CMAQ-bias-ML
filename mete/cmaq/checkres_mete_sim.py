import numpy as np
file = np.load('D:/project/data/mete/cmaq/m_sim_daily.npy')
print(file.shape) #(232, 182, 365, 7) 
#'RH'%,'SFC_TMP'℃,'PBLH'm,'SOL_RAD'W/m2,'precip'cm,'WSPD10'm/s,'WDIR10'deg
mete_name = ['RH','SFC_TMP','PBLH','SOL_RAD','precip','WSPD10','WDIR10']
for i in range(7):
    print(mete_name[i])
    print(np.around(file[136,115,:,i],decimals=2))
