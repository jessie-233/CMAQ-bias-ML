from tensorflow.keras.models import load_model
from sklearn import preprocessing
import numpy as np
import pandas as pd
import copy

np.set_printoptions(suppress=True) #抑制使用对小数的科学记数法
np.set_printoptions(precision=2) #设置输出的精度

data = np.load("D:/project/data/BTH/dataset_abs_corr2.npy") #(44, 363, 42)
all_vars = {'PM2.5_Bias', 'PM10_Bias', 'NO2_Bias', 'SO2_Bias', 'O3_Bias', 'CO_Bias', 'PM2.5_Obs', 'PM10_Obs', 'NO2_Obs', 'SO2_Obs', 'O3_Obs', 'CO_Obs', 'PM2.5_Sim','PM10_Sim','NO2_Sim','SO2_Sim','O3_Sim','CO_Sim', 'RH_Bias', 'TEM_Bias', 'WSPD_Bias', 'WDIR_Bias', 'PRE_Bias', 'RH_Obs', 'TEM_Obs', 'WSPD_Obs', 'WDIR_Obs', 'PRE_Obs', 'PBLH_Sim', 'SOLRAD_Sim','RH_Sim','TEM_Sim','WSPD_Sim','WDIR_Sim','PRE_Sim', 'WIN_N_Obs','WIN_N_Sim', 'WIN_E_Obs','WIN_E_Sim', 'WIN_N_Bias', 'WIN_E_Bias', 'PM2.5_Bias_ystd'}
var_dict = {'PM2.5_Bias':0, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'PM2.5_Obs':6, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'PM2.5_Sim':12, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'WIN_N_Obs':35, 'WIN_E_Obs':37, 'WIN_N_Bias':39, 'WIN_E_Bias':40, 'PM2.5_Bias_ystd':41}
var_sele = ['PM2.5_Bias_ystd','PM2.5_Sim','NO2_Bias','RH_Bias','O3_Bias','SO2_Bias','WSPD_Bias','NO2_Obs','O3_Obs']
region_num = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,2,2,0,0,0,2,0,0,2,2,2,2]) #北京0，天津1，河北2
#取平均
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
print(datasetX_bj.shape) #(363,9)

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
#bj_detail
bj_detail = np.zeros((datasetX_bj.shape[0],datasetX_bj.shape[1]))  
for i in range(datasetX_bj.shape[0]):
    print(i)
    bj_detail[i,:] = Zeroing(datasetX_bj, i, DNN_model)
bj_detail_pd = pd.DataFrame(bj_detail,index=np.arange(1,len(datasetX_bj)+1),columns=var_sele)
#tj_detail
tj_detail = np.zeros((datasetX_tj.shape[0],datasetX_tj.shape[1]))  
for i in range(datasetX_tj.shape[0]):
    print(i)
    tj_detail[i,:] = Zeroing(datasetX_tj, i, DNN_model)
tj_detail_pd = pd.DataFrame(tj_detail,index=np.arange(1,len(datasetX_tj)+1),columns=var_sele)
#hb_detail
hb_detail = np.zeros((datasetX_hb.shape[0],datasetX_hb.shape[1]))  
for i in range(datasetX_hb.shape[0]):
    print(i)
    hb_detail[i,:] = Zeroing(datasetX_hb, i, DNN_model)
hb_detail_pd = pd.DataFrame(hb_detail,index=np.arange(1,len(datasetX_hb)+1),columns=var_sele)

writer = pd.ExcelWriter("D:/project/data/BTH/sensitivity_contribution/contribution_analysis_temp.xlsx")
bj_detail_pd.to_excel(writer,'bj_detail',float_format='%.1f')
tj_detail_pd.to_excel(writer,'tj_detail',float_format='%.1f')
hb_detail_pd.to_excel(writer,'hb_detail',float_format='%.1f')
writer.close()