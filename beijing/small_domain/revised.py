import numpy as np
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from sklearn import metrics
import math

np.random.seed(42)
file = np.load("D:/project/data/beijing/small_domain/dataset_abs_all.npy") #(365,42)
file[0,41] = -30 #根据平均值补全第一条记录
dataset_all = file #(365,42)
dataset_winter = np.concatenate((file[:60,:],file[334:,:]),axis=0) #(91,42)

var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'WIN_N_Obs':35, 'WIN_E_Obs':37, 'WIN_N_Bias':39, 'WIN_E_Bias':40, 'PM2.5_Bias_ystd':41}

# 数据集制作
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','CO_Bias','NO2_Obs','SO2_Obs','O3_Obs','CO_Obs','RH_Bias','TEM_Bias','WIN_N_Bias','WIN_E_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']

def get_xy_dataset(input_dataset):
    global scaler1, scaler2    
    Y = input_dataset[:,0] #'PM2.5_Bias'
    X = np.zeros((len(input_dataset),len(var_sele)))
    i = 0
    for var in var_sele:
        X[:,i] = input_dataset[:,var_dict.get(var)]
        i += 1
    scaler1 = preprocessing.StandardScaler().fit(X)#归一化
    X = scaler1.transform(X)
    Y = Y.reshape((Y.shape[0],1))
    scaler2 = preprocessing.StandardScaler().fit(Y)
    Y = scaler2.transform(Y)
    return X, Y

datasetX_winter, datasetY_winter = get_xy_dataset(dataset_winter)
datasetX_all, datasetY_all = get_xy_dataset(dataset_all)
#print(datasetX_all.shape) #(365, 21)

##计算修正后的PM2.5:bias=sim-obs,PM25_revised=sim-bias，并保存到revised.csv
model = load_model("D:/project/data/beijing/small_domain/DNN_model.h5")
y_pred = model.predict(datasetX_all)
y_pred = scaler2.inverse_transform(y_pred)
PM25_revised = dataset_all[:,var_dict.get('PM2.5_Sim')] - y_pred.reshape(len(datasetX_all),)
PM25_revised = PM25_revised.reshape((len(PM25_revised),-1)) #(365,1)
data = np.concatenate((y_pred,PM25_revised),axis = 1) #(365,2) 第一列是预测的误差值，第二列是修正后的PM2.5
data = np.round(data,2)
np.savetxt("E:/project/beijing/small_domain/revised.csv", data, delimiter=',')


##计算修正后的各项指标
#RMSE
def cal_rmse(xtest, ytest = None, info = ''):
    global model
    if ytest is None:
        RMSE = math.sqrt(metrics.mean_squared_error(xtest[:,var_dict.get('PM2.5_Obs')], xtest[:,var_dict.get('PM2.5_Sim')]))
    else:
        y_pred = model.predict(xtest)
        y_pred = scaler2.inverse_transform(y_pred)
        PM25_revised = ytest[:,var_dict.get('PM2.5_Sim')] - y_pred.reshape(len(xtest),)
        RMSE = math.sqrt(metrics.mean_squared_error(ytest[:,var_dict.get('PM2.5_Obs')], PM25_revised))
    print(info + ' RMSE: %.3f' % RMSE)

print("-----RMSE-----")
cal_rmse(dataset_all, info = 'all year')
cal_rmse(datasetX_all, dataset_all, 'all year revised')
cal_rmse(dataset_winter, info = 'winter')
cal_rmse(datasetX_winter, dataset_winter, 'winter revised')

#R2
def cal_R2(xtest, ytest = None, info = ''):
    global model
    if ytest is None:
        u = np.concatenate((xtest[:,var_dict.get('PM2.5_Obs')], xtest[:,var_dict.get('PM2.5_Sim')])).reshape((2,-1))
        R2 = np.corrcoef(u)[0,1] ** 2
    else:
        y_pred = model.predict(xtest)
        y_pred = scaler2.inverse_transform(y_pred)
        PM25_revised = ytest[:,var_dict.get('PM2.5_Sim')] - y_pred.reshape(len(xtest),)
        v = np.concatenate((ytest[:,var_dict.get('PM2.5_Obs')], PM25_revised)).reshape((2,-1))
        R2 = np.corrcoef(v)[0,1] ** 2
    print(info + ' R2: %.3f' % R2)

print("-----R2-----")
cal_R2(dataset_all, info = 'all year')
cal_R2(datasetX_all, dataset_all, 'all year revised')
cal_R2(dataset_winter, info = 'winter')
cal_R2(datasetX_winter, dataset_winter, 'winter revised')

#NMB
def cal_NMB(xtest, ytest = None, info = ''):
    global model
    if ytest is None:
        NMB = np.sum(xtest[:,var_dict.get('PM2.5_Sim')] - xtest[:,var_dict.get('PM2.5_Obs')]) / np.sum(xtest[:,var_dict.get('PM2.5_Obs')])
    else:
        y_pred = model.predict(xtest)
        y_pred = scaler2.inverse_transform(y_pred)
        #bias=sim-obs,PM25_revised=sim-bias
        PM25_revised = ytest[:,var_dict.get('PM2.5_Sim')] - y_pred.reshape(len(xtest),)
        NMB = np.sum(PM25_revised - ytest[:,var_dict.get('PM2.5_Obs')]) / np.sum(ytest[:,var_dict.get('PM2.5_Obs')])
    print(info + ' NMB: %.3f' % NMB)

print("-----NMB-----")
cal_NMB(dataset_all, info = 'all year')
cal_NMB(datasetX_all, dataset_all, 'all year revised')
cal_NMB(dataset_winter, info = 'winter')
cal_NMB(datasetX_winter, dataset_winter, 'winter revised')

