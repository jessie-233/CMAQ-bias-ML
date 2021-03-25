import numpy as np
np.set_printoptions(precision=2) #输出精度2
np.set_printoptions(suppress=True)#拒绝科学计数法
all_vars = ['PM2.5_Bias', 'PM10_Bias', 'NO2_Bias', 'SO2_Bias', 'O3_Bias', 'CO_Bias', 'PM2.5_Obs', 'PM10_Obs', 'NO2_Obs', 'SO2_Obs', 'O3_Obs', 'CO_Obs', 'PM2.5_Sim','PM10_Sim','NO2_Sim','SO2_Sim','O3_Sim','CO_Sim', 'RH_Bias', 'TEM_Bias', 'WSPD_Bias', 'WDIR_Bias', 'PRE_Bias', 'RH_Obs', 'TEM_Obs', 'WSPD_Obs', 'WDIR_Obs', 'PRE_Obs', 'PBLH_Sim', 'SOLRAD_Sim','RH_Sim','TEM_Sim','WSPD_Sim','WDIR_Sim','PRE_Sim', 'WIN_N_Obs','WIN_N_Sim', 'WIN_E_Obs','WIN_E_Sim', 'WIN_N_Bias', 'WIN_E_Bias', 'PM2.5_Bias_ystd']
var_dict = {'PM2.5_Bias':0, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'PM2.5_Obs':6, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'PM2.5_Sim':12, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'WIN_N_Obs':35, 'WIN_E_Obs':37, 'WIN_N_Bias':39, 'WIN_E_Bias':40, 'PM2.5_Bias_ystd':41}
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WDIR_Bias','WSPD_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']

data = np.load("D:/project/data/BTH/new/dataset_abs_corr2.npy") #(44, 363, 42)
p_loc = [(129,99),(130,99),(139,100),(141,100),(135,101),(141,101),(138,102),(139,102),(141,102),(130,103),(141,103),(145,103),(146,103),(136,105),(141,105),(142,105),(127,106),(134,106),(143,106),(144,106),(127,107),(138,109),(133,111),(139,111),(138,112),(139,112),(140,112),(138,113),(140,113),(137,114),(135,115),(136,115),(141,115),(142,115),(135,116),(136,116),(137,116),(146,116),(135,117),(136,117),(131,119),(140,120),(144,125),(143,126)] #44个
region_num = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,2,2,0,0,0,2,0,0,2,2,2,2]) #北京0，天津1，河北2

# a = data[np.where(region_num == 0),:,:] #(1, 8, 363, 42)
#取平均
low=  [0,-500,-200,-200,-100,0,0,0,-50,-10,0,-5,-5,0,-20,0,0,0,0]
high = [400,200,80,200,100,220,200,200,40,10,180,5,5,100,40,10,9,2500,600]


for i in range(44):
    j = 0
    for var in var_sele:
        for day in range(363):
            a = np.sum((data[i,day,var_dict.get(var)] < low[j]) | (data[i,day,var_dict.get(var)] > high[j]))
            if a != 0:
                print('data['+str(i)+':'+str(day)+':'+var+']')
                print(data[i,day,var_dict.get(var)])
        j += 1
