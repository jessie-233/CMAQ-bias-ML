from tensorflow.keras.models import load_model
from sklearn import preprocessing
import numpy as np
import pandas as pd
#创建excel，北京贡献度'bj-detail'，'bj-overall'，天津、河北 共6个sheets
data = np.load("D:/project/data/BTH/new/dataset_abs.npy") #(44, 363, 42)
all_vars = {'PM2.5_Bias', 'PM10_Bias', 'NO2_Bias', 'SO2_Bias', 'O3_Bias', 'CO_Bias', 'PM2.5_Obs', 'PM10_Obs', 'NO2_Obs', 'SO2_Obs', 'O3_Obs', 'CO_Obs', 'PM2.5_Sim','PM10_Sim','NO2_Sim','SO2_Sim','O3_Sim','CO_Sim', 'RH_Bias', 'TEM_Bias', 'WSPD_Bias', 'WDIR_Bias', 'PRE_Bias', 'RH_Obs', 'TEM_Obs', 'WSPD_Obs', 'WDIR_Obs', 'PRE_Obs', 'PBLH_Sim', 'SOLRAD_Sim','RH_Sim','TEM_Sim','WSPD_Sim','WDIR_Sim','PRE_Sim', 'WIN_N_Obs','WIN_N_Sim', 'WIN_E_Obs','WIN_E_Sim', 'WIN_N_Bias', 'WIN_E_Bias', 'PM2.5_Bias_ystd'}
var_dict = {'PM2.5_Bias':0, 'NO2_Bias':2, 'SO2_Bias':3, 'PM2.5_Obs':6, 'NO2_Obs':8, 'SO2_Obs':9, 'PM2.5_Sim':12, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'WIN_N_Obs':35, 'WIN_E_Obs':37, 'WIN_N_Bias':39, 'WIN_E_Bias':40, 'PM2.5_Bias_ystd':41}
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','NO2_Obs','SO2_Obs','RH_Bias','TEM_Bias','WIN_N_Bias','WIN_E_Bias','PRE_Bias','RH_Obs','TEM_Obs','WIN_N_Obs','WIN_E_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']
region_num = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,2,2,0,0,0,2,0,0,2,2,2,2]) #北京0，天津1，河北2
bj_data = np.mean(data[np.where(region_num == 0),:,:],axis=(0,1)) #(363, 42)
tj_data = np.mean(data[np.where(region_num == 1),:,:],axis=(0,1))
hb_data = np.mean(data[np.where(region_num == 2),:,:],axis=(0,1))

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
print(datasetX_bj.shape) #(363,18)

#贡献度分析（经过实验，delta=0.01时可以使结果稳定下来。）
def get_contribution(datasetX, x_num, NNmodel, delta = 0.01):
    x_sample = datasetX[x_num,:].reshape((-1,datasetX.shape[1])) #已经归一化过的
    y_pred = float(NNmodel.predict(x_sample))
    y = np.zeros(x_sample.shape[1])
    result = np.zeros(x_sample.shape[1])
    for i in range(x_sample.shape[1]):
        x_sample[0,i] = x_sample[0,i] + delta #加delta
        y[i] = float(NNmodel.predict(x_sample))
        x_sample[0,i] = x_sample[0,i] - delta #恢复
        #print("取第{0}条数据，第{1}个变量上调{2}，y增加{3}".format(x_num, i, delta, y[i] - y_pred))
    sum_delt = sum(abs(y - y_pred))
    for i in range(x_sample.shape[1]):
        result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
        #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
    result = np.around(result, decimals=2)
    return result

#加载训练好的model
DNN_model = load_model("D:/project/data/BTH/new/DNN_model.h5")

#check on beijing data
for i in range(datasetX_bj.shape[0]):
    results = np.zeros((3,datasetX_bj.shape[1])) #存储每天的3个delta结果
    num = 0
    print("天数：",i)
    for delta in [0.001,0.005,0.01]:
        #print("delta =",delta)
        result = get_contribution(datasetX_bj, i, DNN_model, delta = delta)
        results[num,:] = result
        num += 1
    dif = sum(abs(results[0]-results[1])) +  sum(abs(results[1]-results[2]))
    if dif < (datasetX_bj.shape[1] * 1 * 2):
        print("day %d pass!" %i)
    else:
        print("day %d fail!" %i)



# print(get_contribution(datasetX_bj, 356, DNN_model)) #除以了0 RuntimeWarning: invalid value encountered in double_scalars
# x_sample = datasetX_bj[356,:].reshape((-1,datasetX_bj.shape[1])) #已经归一化过的 nothing special, but high PM25 bias (y)
# print(x_sample)
# '''
# [[ 0.1027151  -1.64714444 -2.70347879  1.58173829  3.34513664 -0.74338465
#   -3.02800019  0.43040322  0.01624385 -0.01251142 -0.06934564  1.15723819
#   -1.57180115 -0.03577759 -0.03198968 -0.2438569  -1.23752708 -1.00743187]]
# '''
# y_pred = float(DNN_model.predict(x_sample))
# #print(y_pred) #-4.1822590827941895
# y = np.zeros(x_sample.shape[1])
# result = np.zeros(x_sample.shape[1])
# for i in range(x_sample.shape[1]):
#     x_sample[0,i] = x_sample[0,i] + 0.117 #加delta 从0.01往上加，至0.117有值
#     y[i] = float(DNN_model.predict(x_sample))
#     x_sample[0,i] = x_sample[0,i] - 0.117 #恢复
#     #print("取第{0}条数据，第{1}个变量上调{2}，y增加{3}".format(332, i, 0.01, y[i] - y_pred))
# sum_delt = sum(abs(y - y_pred))
# for i in range(x_sample.shape[1]):
#     result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
#     #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
# result = np.around(result, decimals=2)
# print(result)




