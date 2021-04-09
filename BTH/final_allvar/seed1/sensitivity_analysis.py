from tensorflow.keras.models import load_model
from sklearn import preprocessing
import numpy as np
import pandas as pd
import copy

np.set_printoptions(suppress=True) #抑制使用对小数的科学记数法
np.set_printoptions(precision=2) #设置输出的精度

all_vars = ['PM2.5_Bias', 'PM10_Bias', 'NO2_Bias', 'SO2_Bias', 'O3_Bias', 'CO_Bias', 'PM2.5_Obs', 'PM10_Obs', 'NO2_Obs', 'SO2_Obs', 'O3_Obs', 'CO_Obs', 'PM2.5_Sim','PM10_Sim','NO2_Sim','SO2_Sim','O3_Sim','CO_Sim', 'RH_Bias', 'TEM_Bias', 'WSPD_Bias', 'WDIR_Bias', 'PRE_Bias', 'RH_Obs', 'TEM_Obs', 'WSPD_Obs', 'WDIR_Obs', 'PRE_Obs', 'PBLH_Sim', 'SOLRAD_Sim','RH_Sim','TEM_Sim','WSPD_Sim','WDIR_Sim','PRE_Sim','PM2.5_Bias_ystd','NO2_Bias_ystd','RH_Bias_ystd','O3_Bias_ystd','SO2_Bias_ystd','WSPD_Bias_ystd','NO2_Obs_ystd','O3_Obs_ystd']
var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'PM10_Sim':13, 'NO2_Sim':14, 'SO2_Sim':15, 'O3_Sim':16, 'CO_Sim':17, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'RH_Sim':30, 'TEM_Sim':31, 'WSPD_Sim':32, 'WDIR_Sim':33, 'PRE_Sim':34, 'PM2.5_Bias_ystd':35, 'NO2_Bias_ystd':36, 'RH_Bias_ystd':37, 'O3_Bias_ystd':38, 'SO2_Bias_ystd':39, 'WSPD_Bias_ystd':40, 'NO2_Obs_ystd':41, 'O3_Obs_ystd':42}
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WDIR_Bias','WSPD_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']
data = np.load("D:/project/data/BTH/dataset_BTH.npy") #(44, 363, 42)
dataset_all = data.reshape((-1,43))
#取平均
BTH_data = np.mean(data,axis=0) #(363, 43)
#计算scaler1, scaler2
Y_all = dataset_all[:,0].reshape((dataset_all.shape[0],1))
X_all = np.zeros((len(dataset_all),len(var_sele)))
i = 0
for var in var_sele:
    X_all[:,i] = dataset_all[:,var_dict.get(var)]
    i += 1
scaler1 = preprocessing.StandardScaler().fit(X_all)
scaler2 = preprocessing.StandardScaler().fit(Y_all)

#prepare x & y dataset
def get_xy_dataset(input_dataset):
    global scaler1, scaler2    
    Y = input_dataset[:,0] #'PM2.5_Bias'
    X = np.zeros((len(input_dataset),len(var_sele)))
    i = 0
    for var in var_sele:
        X[:,i] = input_dataset[:,var_dict.get(var)]
        i += 1
    X = scaler1.transform(X) #标准化
    Y = Y.reshape((Y.shape[0],1))
    Y = scaler2.transform(Y) #标准化
    return X, Y

datasetX, datasetY = get_xy_dataset(BTH_data)
print(datasetX.shape) #(363,9)
print(datasetY.shape) #(363,1)

DNN_model = load_model("D:/project/data/BTH/final_allvar/seed1/DNN_BTH_1.h5")

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

#制作sheets
#BTH_detail
BTH_detail = np.zeros((datasetX.shape[0],datasetX.shape[1]))  
for i in range(datasetX.shape[0]):
    print(i)
    BTH_detail[i,:] = get_contribution(datasetX, i, DNN_model)
BTH_detail_pd = pd.DataFrame(np.concatenate((BTH_detail,BTH_data[:,var_dict.get('PM2.5_Obs')]),axis=1),index=np.arange(1,len(datasetX)+1),columns=var_sele)

#BTH_combine
BTH_combine_pd = pd.DataFrame({'PM2.5_Bias_ystd':BTH_detail_pd.loc[:,'PM2.5_Bias_ystd'],
                               'Pollutants':BTH_detail_pd.loc[:,'PM2.5_Sim']+BTH_detail_pd.loc[:,'NO2_Obs']+BTH_detail_pd.loc[:,'NO2_Bias']+BTH_detail_pd.loc[:,'SO2_Bias']+BTH_detail_pd.loc[:,'SO2_Obs'],
                               'Wind':BTH_detail_pd.loc[:,'WSPD_Bias']+BTH_detail_pd.loc[:,'WSPD_Obs']+BTH_detail_pd.loc[:,'WDIR_Bias'],
                               'Precipitation':BTH_detail_pd.loc[:,'PRE_Bias']+BTH_detail_pd.loc[:,'PRE_Obs'],
                               'Weather Condition':BTH_detail_pd.loc[:,'O3_Obs']+BTH_detail_pd.loc[:,'O3_Bias']+BTH_detail_pd.loc[:,'TEM_Obs']+BTH_detail_pd.loc[:,'TEM_Bias']+BTH_detail_pd.loc[:,'SOLRAD_Sim']+BTH_detail_pd.loc[:,'PBLH_Sim']+BTH_detail_pd.loc[:,'RH_Obs']+BTH_detail_pd.loc[:,'RH_Bias'],
                               'PM2.5_Obs':BTH_data[:,var_dict.get('PM2.5_Obs')]},index = np.arange(1,len(datasetX)+1))

#写入excel
writer = pd.ExcelWriter("D:/project/data/BTH/final_allvar/seed1/sensitivity_analysis_temp.xlsx")
BTH_detail_pd.to_excel(writer,'BTH_detail',float_format='%.1f')
BTH_combine_pd.to_excel(writer,'BTH_combine',float_format='%.1f')
writer.close()

# #check异常值：sum_delta=0，说明delta太小
# #datasetX_bj[356,:]
# print(get_contribution(datasetX_bj, 356, DNN_model)) #除以了0 RuntimeWarning: invalid value encountered in double_scalars
# x_sample = datasetX_bj[356,:].reshape((-1,datasetX_bj.shape[1])) #high NO2_Obs,low RH_Bias,high in PM2.5_Bias&PM2.5_Obs
# print(x_sample)

# # [[ 0.1  -1.65 -2.7   1.58  0.04  3.35 -0.74 -1.32 -3.03  0.43  0.54 -0.19
# #   -0.07  1.16 -1.57 -0.57 -0.24 -1.24 -1.01]]

# y_pred = float(DNN_model.predict(x_sample))
# y = np.zeros(x_sample.shape[1])
# result = np.zeros(x_sample.shape[1])
# for i in range(x_sample.shape[1]):
#     x_sample[0,i] = x_sample[0,i] + 0.5 #加delta 从0.01往上加，至0.5有值，倒数第三个变量贡献度100%
#     y[i] = float(DNN_model.predict(x_sample))
#     x_sample[0,i] = x_sample[0,i] - 0.5 #恢复
#     #print("取第{0}条数据，第{1}个变量上调{2}，y增加{3}".format(332, i, 0.01, y[i] - y_pred))
# sum_delt = sum(abs(y - y_pred))
# for i in range(x_sample.shape[1]):
#     result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
#     #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
# result = np.around(result, decimals=2)
# print(result)




