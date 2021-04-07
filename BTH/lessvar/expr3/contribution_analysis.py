from tensorflow.keras.models import load_model
from sklearn import preprocessing
import numpy as np
import pandas as pd
import copy

np.set_printoptions(suppress=True) #抑制使用对小数的科学记数法
np.set_printoptions(precision=2) #设置输出的精度

data = np.load("D:/project/data/BTH/DOMAIN_TRANS/BTH/dataset_BTH.npy") #(44,363,43)
dataset_all = data.reshape((-1,43))
var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'PM10_Sim':13, 'NO2_Sim':14, 'SO2_Sim':15, 'O3_Sim':16, 'CO_Sim':17, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'RH_Sim':30, 'TEM_Sim':31, 'WSPD_Sim':32, 'WDIR_Sim':33, 'PRE_Sim':34, 'PM2.5_Bias_ystd':35, 'NO2_Bias_ystd':36, 'RH_Bias_ystd':37, 'O3_Bias_ystd':38, 'SO2_Bias_ystd':39, 'WSPD_Bias_ystd':40, 'NO2_Obs_ystd':41, 'O3_Obs_ystd':42}
var_sele = ['PM2.5_Bias_ystd','PM2.5_Sim','NO2_Bias','RH_Bias','SO2_Bias','WSPD_Bias','NO2_Obs','TEM_Obs','RH_Obs']

BTH_data = np.mean(data,axis=0) #(363, 43)

#standardization: scaler of dataset_all
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

DNN_model = load_model("D:/project/data/BTH/lessvar/expr3/DNN_model3.h5")

'''
#方法一：分别看每个变量，将其他所有变量置零，打印revised
def Choosing(day_num):
    global datasetX_bj,bj_data,scaler2
    x_sample = datasetX_bj[day_num,:].reshape((-1,datasetX_bj.shape[1])) #(1,9)
    sim = bj_data[day_num,var_dict.get('PM2.5_Sim')]
    obs = bj_data[day_num,var_dict.get('PM2.5_Obs')]
    revised = bj_data[day_num,var_dict.get('PM2.5_Sim')] - float(scaler2.inverse_transform(model.predict(x_sample)))
    print('-------original for day {0}-------'.format(day_num))
    print('sim PM:',sim)
    print('obs PM:',obs)
    print('overall revised PM:',revised)
    print('-------experiment for day {0}-------'.format(day_num))
    temp = copy.deepcopy(x_sample)
    for i in range(datasetX_bj.shape[1]):
        for j in range(datasetX_bj.shape[1]):
            if j != i:
                temp[:,j] = 0
        bias_pred = float(scaler2.inverse_transform(model.predict(temp)))
        #print('for var {0}, predicted bias is:{1}:'.format(i,bias_pred))
        revised = bj_data[0,var_dict.get('PM2.5_Sim')] - bias_pred
        temp = copy.deepcopy(x_sample) #恢复
        print('for var {0}, revised PM is:{1}'.format(i,revised))

#方法二：所有变量置零，一个个加入变量，打印revised
def Adding(day_num):
    global datasetX_bj,bj_data,scaler2
    x_sample = datasetX_bj[day_num,:].reshape((-1,datasetX_bj.shape[1])) #(1,18)
    sim = bj_data[day_num,var_dict.get('PM2.5_Sim')]
    obs = bj_data[day_num,var_dict.get('PM2.5_Obs')]
    revised = bj_data[day_num,var_dict.get('PM2.5_Sim')] - float(scaler2.inverse_transform(model.predict(x_sample)))
    print('-------original for day {0}-------'.format(day_num))
    print('sim PM:',sim)
    print('obs PM:',obs)
    print('overall revised PM:',revised)
    print('-------experiment for day {0}-------'.format(day_num))
    temp = np.zeros((1,datasetX_bj.shape[1]))
    bias_pred = float(scaler2.inverse_transform(model.predict(temp)))
    revised = bj_data[0,var_dict.get('PM2.5_Sim')] - bias_pred
    print('with all zeros,revised PM is:{0}'.format(revised))
    for i in range(datasetX_bj.shape[1]):
        temp[:,i] = x_sample[:,i]
        bias_pred = float(scaler2.inverse_transform(model.predict(temp)))
        #print('for var {0}, predicted bias is:{1}:'.format(i,bias_pred))
        revised = bj_data[0,var_dict.get('PM2.5_Sim')] - bias_pred
        print('after adding var {0}, revised PM is:{1}'.format(i,revised))
'''
#方法三：分别将每个变量置零(平均)
def Zeroing(datasetX, x_num, NNmodel, delta = 0.01):
    x_sample = datasetX[x_num,:].reshape((-1,datasetX.shape[1])) #已经归一化过的
    y_pred = float(NNmodel.predict(x_sample))
    y = np.zeros(x_sample.shape[1])
    result = np.zeros(x_sample.shape[1])
    temp = copy.deepcopy(x_sample)
    for i in range(x_sample.shape[1]):
        temp = copy.deepcopy(x_sample[0,i])
        x_sample[0,i] = 0 #加delta
        y[i] = float(NNmodel.predict(x_sample))
        x_sample[0,i] = temp #恢复
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
    BTH_detail[i,:] = Zeroing(datasetX, i, DNN_model)
BTH_detail_pd = pd.DataFrame(BTH_detail,index=np.arange(1,len(datasetX)+1),columns=var_sele)

#写入excel
writer = pd.ExcelWriter("D:/project/data/BTH/lessvar/expr3/contribution_analysis_temp.xlsx")
BTH_detail_pd.to_excel(writer,'BTH_detail',float_format='%.1f')
writer.close()