import numpy as np
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from sklearn import metrics
import math
import pandas as pd
all_vars = ['PM2.5_Bias', 'PM10_Bias', 'NO2_Bias', 'SO2_Bias', 'O3_Bias', 'CO_Bias', 'PM2.5_Obs', 'PM10_Obs', 'NO2_Obs', 'SO2_Obs', 'O3_Obs', 'CO_Obs', 'PM2.5_Sim','PM10_Sim','NO2_Sim','SO2_Sim','O3_Sim','CO_Sim', 'RH_Bias', 'TEM_Bias', 'WSPD_Bias', 'WDIR_Bias', 'PRE_Bias', 'RH_Obs', 'TEM_Obs', 'WSPD_Obs', 'WDIR_Obs', 'PRE_Obs', 'PBLH_Sim', 'SOLRAD_Sim','RH_Sim','TEM_Sim','WSPD_Sim','WDIR_Sim','PRE_Sim','PM2.5_Bias_ystd','NO2_Bias_ystd','RH_Bias_ystd','O3_Bias_ystd','SO2_Bias_ystd','WSPD_Bias_ystd','NO2_Obs_ystd','O3_Obs_ystd']
var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'PM10_Sim':13, 'NO2_Sim':14, 'SO2_Sim':15, 'O3_Sim':16, 'CO_Sim':17, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'RH_Sim':30, 'TEM_Sim':31, 'WSPD_Sim':32, 'WDIR_Sim':33, 'PRE_Sim':34, 'PM2.5_Bias_ystd':35, 'NO2_Bias_ystd':36, 'RH_Bias_ystd':37, 'O3_Bias_ystd':38, 'SO2_Bias_ystd':39, 'WSPD_Bias_ystd':40, 'NO2_Obs_ystd':41, 'O3_Obs_ystd':42}
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WDIR_Bias','WSPD_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']
data = np.load("D:/project/data/YRD/dataset_YRD.npy") #(44, 363, 42)

#取平均
YRD_data = np.mean(data,axis=0) #(363, 43)
YRD_data_pd = pd.DataFrame(YRD_data,index=np.arange(1,364),columns=all_vars)#(363, 42)
writer = pd.ExcelWriter("D:/project/data/YRD/final_allvar/seed1/decriptive_analysis_temp.xlsx")
YRD_data_pd.to_excel(writer,'YRD_basic',float_format='%.2f')	#保留小数点后2位

#计算scaler1, scaler2
data = data.reshape((-1,43))
X = np.zeros((len(data),len(var_sele))) #(15972, 19)
Y = data[:,0].reshape((len(data),1)) #(15972, 1)
i = 0
for var in var_sele:
    X[:,i] = data[:,var_dict.get(var)]
    i += 1
scaler1 = preprocessing.StandardScaler().fit(X)
scaler2 = preprocessing.StandardScaler().fit(Y)


def get_xy_dataset(input_dataset):
    global scaler1, scaler2    
    Y = input_dataset[:,0] #'PM2.5_Bias'
    X = np.zeros((len(input_dataset),len(var_sele)))
    i = 0
    for var in var_sele:
        X[:,i] = input_dataset[:,var_dict.get(var)]
        i += 1
    X = scaler1.transform(X)
    Y = Y.reshape((Y.shape[0],1))
    Y = scaler2.transform(Y)
    return X, Y

datasetX, datasetY = get_xy_dataset(YRD_data)

##计算修正后的PM2.5:bias=sim-obs,PM25_revised=sim-bias,PM25_Bias_revised=revised-obs
model = load_model("D:/project/data/YRD/final_allvar/seed1/DNN_YRD_1.h5")
y_pred = model.predict(datasetX)
y_pred = scaler2.inverse_transform(y_pred).reshape(len(datasetX),) #(363,)
PM25_revised = YRD_data[:,var_dict.get('PM2.5_Sim')] - y_pred
PM25_Bias_revised = PM25_revised - YRD_data[:,var_dict.get('PM2.5_Obs')]

#YRD_PM25 sheet：第1列PM2.5_Obs，第2列PM2.5_Sim，第3列PM2.5_Bias，第4列PM2.5_Bias_Predict，第5列PM2.5_revised，第6列PM2.5_Bias_revised
YRD_pm = pd.DataFrame({'PM2.5_Obs':YRD_data[:,6],'PM2.5_Sim':YRD_data[:,12],'PM2.5_Bias':YRD_data[:,0],'PM2.5_Bias_Predict':y_pred,'PM2.5_revised':PM25_revised,'PM2.5_Bias_revised':PM25_Bias_revised},index=np.arange(1,364))
YRD_pm.to_excel(writer, 'YRD_PM25', float_format='%.2f')	#保留小数点后2位
writer.close()