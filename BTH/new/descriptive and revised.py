import numpy as np
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from sklearn import metrics
import math
import pandas as pd
all_vars = ['PM2.5_Bias', 'PM10_Bias', 'NO2_Bias', 'SO2_Bias', 'O3_Bias', 'CO_Bias', 'PM2.5_Obs', 'PM10_Obs', 'NO2_Obs', 'SO2_Obs', 'O3_Obs', 'CO_Obs', 'PM2.5_Sim','PM10_Sim','NO2_Sim','SO2_Sim','O3_Sim','CO_Sim', 'RH_Bias', 'TEM_Bias', 'WSPD_Bias', 'WDIR_Bias', 'PRE_Bias', 'RH_Obs', 'TEM_Obs', 'WSPD_Obs', 'WDIR_Obs', 'PRE_Obs', 'PBLH_Sim', 'SOLRAD_Sim','RH_Sim','TEM_Sim','WSPD_Sim','WDIR_Sim','PRE_Sim', 'WIN_N_Obs','WIN_N_Sim', 'WIN_E_Obs','WIN_E_Sim', 'WIN_N_Bias', 'WIN_E_Bias', 'PM2.5_Bias_ystd']
var_dict = {'PM2.5_Bias':0, 'NO2_Bias':2, 'SO2_Bias':3, 'PM2.5_Obs':6, 'NO2_Obs':8, 'SO2_Obs':9, 'PM2.5_Sim':12, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'WIN_N_Obs':35, 'WIN_E_Obs':37, 'WIN_N_Bias':39, 'WIN_E_Bias':40, 'PM2.5_Bias_ystd':41}
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','NO2_Obs','SO2_Obs','RH_Bias','TEM_Bias','WIN_N_Bias','WIN_E_Bias','PRE_Bias','RH_Obs','TEM_Obs','WIN_N_Obs','WIN_E_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']

data = np.load("D:/project/data/BTH/new/dataset_abs.npy") #(44, 363, 42)
p_loc = [(129,99),(130,99),(139,100),(141,100),(135,101),(141,101),(138,102),(139,102),(141,102),(130,103),(141,103),(145,103),(146,103),(136,105),(141,105),(142,105),(127,106),(134,106),(143,106),(144,106),(127,107),(138,109),(133,111),(139,111),(138,112),(139,112),(140,112),(138,113),(140,113),(137,114),(135,115),(136,115),(141,115),(142,115),(135,116),(136,116),(137,116),(146,116),(135,117),(136,117),(131,119),(140,120),(144,125),(143,126)] #44个
region_num = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,2,2,0,0,0,2,0,0,2,2,2,2]) #北京0，天津1，河北2
# a = data[np.where(region_num == 0),:,:] #(1, 8, 363, 42)
bj_data = np.mean(data[np.where(region_num == 0),:,:],axis=(0,1)) #(363, 42)
tj_data = np.mean(data[np.where(region_num == 1),:,:],axis=(0,1))
hb_data = np.mean(data[np.where(region_num == 2),:,:],axis=(0,1))


bj_data_pd = pd.DataFrame(bj_data,index=np.arange(1,364),columns=all_vars)#(363, 42)
tj_data_pd = pd.DataFrame(tj_data,index=np.arange(1,364),columns=all_vars)
hb_data_pd = pd.DataFrame(hb_data,index=np.arange(1,364),columns=all_vars)

writer = pd.ExcelWriter("D:/project/data/BTH/new/decriptive_analysis.xlsx")
bj_data_pd.to_excel(writer, 'bj', float_format='%.2f')	#保留小数点后2位
tj_data_pd.to_excel(writer, 'tj', float_format='%.2f')
hb_data_pd.to_excel(writer, 'hb', float_format='%.2f')


#计算scaler1, scaler2
data = data.reshape((-1,42)) #(16016, 42)
X = np.zeros((len(data),len(var_sele)))
Y = data[:,0].reshape((len(data),1))
i = 0
for var in var_sele:
    X[:,i] = data[:,var_dict.get(var)]
    i += 1
scaler1 = preprocessing.StandardScaler().fit(X)
scaler2 = preprocessing.StandardScaler().fit(Y)
print(scaler1,scaler2)

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

datasetX_bj, datasetY_bj = get_xy_dataset(bj_data)
datasetX_tj, datasetY_tj = get_xy_dataset(tj_data)
datasetX_hb, datasetY_hb = get_xy_dataset(hb_data)
print(datasetX_bj.shape) #(363,17)

##计算修正后的PM2.5:bias=sim-obs,PM25_revised=sim-bias,PM25_Bias_revised=revised-obs
model = load_model("D:/project/data/BTH/new/DNN_model.h5")
y_pred_bj = model.predict(datasetX_bj)
y_pred_bj = scaler2.inverse_transform(y_pred_bj).reshape(len(datasetX_bj),) #(363,)
print(y_pred_bj.shape)
PM25_revised_bj = bj_data[:,var_dict.get('PM2.5_Sim')] - y_pred_bj
PM25_Bias_revised_bj = PM25_revised_bj - bj_data[:,var_dict.get('PM2.5_Obs')]

y_pred_tj = model.predict(datasetX_tj)
y_pred_tj = scaler2.inverse_transform(y_pred_tj).reshape(len(datasetX_tj),) #(363,)
print(y_pred_tj.shape)
PM25_revised_tj = tj_data[:,var_dict.get('PM2.5_Sim')] - y_pred_tj
PM25_Bias_revised_tj = PM25_revised_tj - tj_data[:,var_dict.get('PM2.5_Obs')]

y_pred_hb = model.predict(datasetX_hb)
y_pred_hb = scaler2.inverse_transform(y_pred_hb).reshape(len(datasetX_hb),) #(363,)
print(y_pred_hb.shape)
PM25_revised_hb = hb_data[:,var_dict.get('PM2.5_Sim')] - y_pred_hb
PM25_Bias_revised_hb = PM25_revised_hb - hb_data[:,var_dict.get('PM2.5_Obs')]


#bj-PM25 sheet：第1列PM2.5_Obs，第2列PM2.5_Sim，第3列PM2.5_Bias，第4列PM2.5_Bias_Predict，第5列PM2.5_revised，第6列PM2.5_Bias_revised
bj_pm = pd.DataFrame({'PM2.5_Obs':bj_data[:,6],'PM2.5_Sim':bj_data[:,12],'PM2.5_Bias':bj_data[:,0],'PM2.5_Bias_Predict':y_pred_bj,'PM2.5_revised':PM25_revised_bj,'PM2.5_Bias_revised':PM25_Bias_revised_bj},index=np.arange(1,364))
tj_pm = pd.DataFrame({'PM2.5_Obs':tj_data[:,6],'PM2.5_Sim':tj_data[:,12],'PM2.5_Bias':tj_data[:,0],'PM2.5_Bias_Predict':y_pred_tj,'PM2.5_revised':PM25_revised_tj,'PM2.5_Bias_revised':PM25_Bias_revised_tj},index=np.arange(1,364))
hb_pm = pd.DataFrame({'PM2.5_Obs':hb_data[:,6],'PM2.5_Sim':hb_data[:,12],'PM2.5_Bias':hb_data[:,0],'PM2.5_Bias_Predict':y_pred_hb,'PM2.5_revised':PM25_revised_hb,'PM2.5_Bias_revised':PM25_Bias_revised_hb},index=np.arange(1,364))
bj_pm.to_excel(writer, 'bj_PM25', float_format='%.2f')	#保留小数点后2位
tj_pm.to_excel(writer, 'tj_PM25', float_format='%.2f')	#保留小数点后2位
hb_pm.to_excel(writer, 'hb_PM25', float_format='%.2f')	#保留小数点后2位
writer.close()