import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imp
foo = imp.load_source('windrose', 'D:/Anaconda/Lib/site-packages/windrose/windrose.py')
from windrose import WindroseAxes
import matplotlib.cm as cm
all_vars = ['PM2.5_Bias', 'PM10_Bias', 'NO2_Bias', 'SO2_Bias', 'O3_Bias', 'CO_Bias', 'PM2.5_Obs', 'PM10_Obs', 'NO2_Obs', 'SO2_Obs', 'O3_Obs', 'CO_Obs', 'PM2.5_Sim','PM10_Sim','NO2_Sim','SO2_Sim','O3_Sim','CO_Sim', 'RH_Bias', 'TEM_Bias', 'WSPD_Bias', 'WDIR_Bias', 'PRE_Bias', 'RH_Obs', 'TEM_Obs', 'WSPD_Obs', 'WDIR_Obs', 'PRE_Obs', 'PBLH_Sim', 'SOLRAD_Sim','RH_Sim','TEM_Sim','WSPD_Sim','WDIR_Sim','PRE_Sim', 'WIN_N_Obs','WIN_N_Sim', 'WIN_E_Obs','WIN_E_Sim', 'WIN_N_Bias', 'WIN_E_Bias', 'PM2.5_Bias_ystd']
var_dict = {'PM2.5_Bias':0, 'NO2_Bias':2, 'SO2_Bias':3, 'PM2.5_Obs':6, 'NO2_Obs':8, 'SO2_Obs':9, 'PM2.5_Sim':12, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'WIN_N_Obs':35, 'WIN_E_Obs':37, 'WIN_N_Bias':39, 'WIN_E_Bias':40, 'PM2.5_Bias_ystd':41}
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','NO2_Obs','SO2_Obs','RH_Bias','TEM_Bias','WIN_N_Bias','WIN_E_Bias','PRE_Bias','RH_Obs','TEM_Obs','WIN_N_Obs','WIN_E_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']

data = np.load("D:/project/data/BTH/new/dataset_abs.npy") #(44, 363, 42) WSPD_Obs：25 WDIR_Obs：26 WSPD_Sim：32 WDIR_Sim：33 PM2.5_Obs：6
region_num = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,2,2,0,0,0,2,0,0,2,2,2,2]) #北京0，天津1，河北2

column = ['WSPD_Obs','WDIR_Obs','WSPD_Sim','WDIR_Sim','PM25_Obs']
#beijing(2904, 5) 第一列WSPD_Obs 第二列WDIR_Obs 第三列WSPD_Sim 第四列WDIR_Sim 第五列PM2.5_Obs
# a = data[np.where(region_num == 0),:,:] #(1, 8, 363, 42)
bj_data = data[np.where(region_num == 0),:,25].reshape((363*8,1))
bj_data = np.concatenate((bj_data,data[np.where(region_num == 0),:,26].reshape((363*8,1))),axis=1)
bj_data = np.concatenate((bj_data,data[np.where(region_num == 0),:,32].reshape((363*8,1))),axis=1)
bj_data = np.concatenate((bj_data,data[np.where(region_num == 0),:,33].reshape((363*8,1))),axis=1)
bj_data = np.concatenate((bj_data,data[np.where(region_num == 0),:,6].reshape((363*8,1))),axis=1)
bj_data = pd.DataFrame(bj_data,columns=column)
#tianjin(2178, 5)
# a = data[np.where(region_num == 1),:,:]  #(1, 6, 363, 42)
tj_data = data[np.where(region_num == 1),:,25].reshape((363*6,1))
tj_data = np.concatenate((tj_data,data[np.where(region_num == 1),:,26].reshape((363*6,1))),axis=1)
tj_data = np.concatenate((tj_data,data[np.where(region_num == 1),:,32].reshape((363*6,1))),axis=1)
tj_data = np.concatenate((tj_data,data[np.where(region_num == 1),:,33].reshape((363*6,1))),axis=1)
tj_data = np.concatenate((tj_data,data[np.where(region_num == 1),:,6].reshape((363*6,1))),axis=1)
tj_data = pd.DataFrame(tj_data,columns=column)
#hebei(10890, 5)
# a = data[np.where(region_num == 1),:,:]  #(1, 30, 363, 42)
hb_data = data[np.where(region_num == 2),:,25].reshape((363*30,1))
hb_data = np.concatenate((hb_data,data[np.where(region_num == 2),:,26].reshape((363*30,1))),axis=1)
hb_data = np.concatenate((hb_data,data[np.where(region_num == 2),:,32].reshape((363*30,1))),axis=1)
hb_data = np.concatenate((hb_data,data[np.where(region_num == 2),:,33].reshape((363*30,1))),axis=1)
hb_data = np.concatenate((hb_data,data[np.where(region_num == 2),:,6].reshape((363*30,1))),axis=1)
hb_data = pd.DataFrame(hb_data,columns=column)


bj_data_1 = bj_data[bj_data.PM25_Obs < 75]
bj_data_2 = bj_data[(bj_data.PM25_Obs >= 75) & (bj_data.PM25_Obs < 115)]
bj_data_3 = bj_data[(bj_data.PM25_Obs >= 115) & (bj_data.PM25_Obs < 150)]
bj_data_4 = bj_data[(bj_data.PM25_Obs >= 150) & (bj_data.PM25_Obs < 250)]
bj_data_5 = bj_data[bj_data.PM25_Obs >= 250]

tj_data_1 = tj_data[tj_data.PM25_Obs < 75]
tj_data_2 = tj_data[(tj_data.PM25_Obs >= 75) & (tj_data.PM25_Obs < 115)]
tj_data_3 = tj_data[(tj_data.PM25_Obs >= 115) & (tj_data.PM25_Obs < 150)]
tj_data_4 = tj_data[(tj_data.PM25_Obs >= 150) & (tj_data.PM25_Obs < 250)]
tj_data_5 = tj_data[tj_data.PM25_Obs >= 250]

hb_data_1 = hb_data[hb_data.PM25_Obs < 75]
hb_data_2 = hb_data[(hb_data.PM25_Obs >= 75) & (hb_data.PM25_Obs < 115)]
hb_data_3 = hb_data[(hb_data.PM25_Obs >= 115) & (hb_data.PM25_Obs < 150)]
hb_data_4 = hb_data[(hb_data.PM25_Obs >= 150) & (hb_data.PM25_Obs < 250)]
hb_data_5 = hb_data[hb_data.PM25_Obs >= 250]

dataset = [bj_data_1,bj_data_2,bj_data_3,bj_data_4,bj_data_5,tj_data_1,tj_data_2,tj_data_3,tj_data_4,tj_data_5,hb_data_1,hb_data_2,hb_data_3,hb_data_4,hb_data_5]
name = ['beijing_1','beijing_2','beijing_3','beijing_4','beijing_5','tianjin_1','tianjin_2','tianjin_3','tianjin_4','tianjin_5','hebei_1','hebei_2','hebei_3','hebei_4','hebei_5']
def Drawwindrose():
    fig = plt.figure(figsize=(50,50))
    fig.tight_layout()
    j = 0
    for i in np.arange(1,30,2): #i = 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29
        #obs
        ax = fig.add_subplot(3, 10, i,projection='windrose')
        ax.contourf(dataset[j].WDIR_Obs, dataset[j].WSPD_Obs, bins=np.arange(0,6,1), cmap=cm.hot) #填充色
        ax.contour(dataset[j].WDIR_Obs, dataset[j].WSPD_Obs, bins=np.arange(0,6,1), colors = 'black', lw=1) #边线
        ax.set_title(name[j]+'_Obs')
        #sim
        ax = fig.add_subplot(3, 10, i+1,projection='windrose')
        ax.contourf(dataset[j].WDIR_Sim, dataset[j].WSPD_Sim, bins=np.arange(0,6,1), cmap=cm.hot) #填充色
        ax.contour(dataset[j].WDIR_Sim, dataset[j].WSPD_Sim, bins=np.arange(0,6,1), colors = 'black', lw=1) #边线
        ax.set_title(name[j]+'_Sim')
        j += 1
    ax.set_legend()
    fig.subplots_adjust()
    #plt.savefig("D:/project/data/BTH/new/windrose.png",bbox_inches='tight')
    plt.show()

Drawwindrose()

