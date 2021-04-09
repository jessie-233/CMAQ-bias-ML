import numpy as np
import pandas as pd
from sklearn import preprocessing
np.set_printoptions(suppress=True) #抑制使用对小数的科学记数法
np.set_printoptions(precision=2) #设置输出的精度

data = np.load("D:/project/data/YRD/dataset_YRD.npy") #(44,363,43)
dataset_all = data.reshape((-1,43))


var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'PM10_Sim':13, 'NO2_Sim':14, 'SO2_Sim':15, 'O3_Sim':16, 'CO_Sim':17, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'RH_Sim':30, 'TEM_Sim':31, 'WSPD_Sim':32, 'WDIR_Sim':33, 'PRE_Sim':34, 'PM2.5_Bias_ystd':35, 'NO2_Bias_ystd':36, 'RH_Bias_ystd':37, 'O3_Bias_ystd':38, 'SO2_Bias_ystd':39, 'WSPD_Bias_ystd':40, 'NO2_Obs_ystd':41, 'O3_Obs_ystd':42}


# NO2_Bias SO2_Bias O3_Bias      RH_Bias TEM_Bias WDIR_Bias WSPD_Bias PRE_Bias
# NO2_Obs SO2_Obs O3_Obs         RH_Obs  TEM_Obs WSPD_Obs PRE_Obs PBLH_Sim SOLRAD_Sim
var_sele = ['PM2.5_Bias','PM2.5_Bias_ystd','PM2.5_Sim','NO2_Bias','SO2_Bias','O3_Bias','RH_Bias','TEM_Bias','WDIR_Bias','WSPD_Bias','PRE_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']
#standardization: scaler of dataset_all
# Y_all = dataset_all[:,0].reshape((dataset_all.shape[0],1))
X_all = np.zeros((len(dataset_all),len(var_sele)))
i = 0
for var in var_sele:
    X_all[:,i] = dataset_all[:,var_dict.get(var)]
    i += 1
scaler1 = preprocessing.StandardScaler().fit(X_all)
# scaler2 = preprocessing.StandardScaler().fit(Y_all)
X_all = scaler1.transform(X_all) #(15972, 10)
print(X_all)
# Y_all = scaler2.transform(Y_all) #(15972, 1)
 
data_df = pd.DataFrame(X_all,columns=var_sele)
cor = data_df.corr() # 计算相关系数，得到一个矩阵
print(cor)
writer = pd.ExcelWriter('D:/project/data/YRD/feature_selection/correlation_YRD_temp.xlsx')
cor.to_excel(writer,float_format='%.2f')
writer.close()
