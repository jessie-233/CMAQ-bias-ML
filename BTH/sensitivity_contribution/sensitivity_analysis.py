from tensorflow.keras.models import load_model
from sklearn import preprocessing
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True) #抑制使用对小数的科学记数法
np.set_printoptions(precision=2) #设置输出的精度
#创建excel，BTH贡献度'BTH-detail'
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
DNN_model = load_model("D:/project/data/BTH/lessvar/expr2/DNN_model2.h5")

#制作sheets
#BTH-detail
BTH_detail = np.zeros((datasetX.shape[0],datasetX.shape[1]))  
for i in range(datasetX.shape[0]):
    print(i)
    BTH_detail[i,:] = get_contribution(datasetX, i, DNN_model)
BTH_detail_pd = pd.DataFrame(BTH_detail,index=np.arange(1,len(datasetX)+1),columns=var_sele)

#写入excel
writer = pd.ExcelWriter("D:/project/data/BTH/lessvar/expr2/sensitivity_analysis_temp.xlsx")
BTH_detail_pd.to_excel(writer,'BTH_detail',float_format='%.1f')
writer.close()

'''
#check异常值：sum_delta=0，说明delta太小
#定义函数
def Check(datasetX,day_num,delta):
    global DNN_model
    print(get_contribution(datasetX, day_num, DNN_model)) #confirm the problem
    x_sample = datasetX[day_num,:].reshape((-1,datasetX.shape[1]))
    print(x_sample) #check the source data and detect the problem
    y_pred = float(DNN_model.predict(x_sample))
    y = np.zeros(x_sample.shape[1])
    result = np.zeros(x_sample.shape[1])
    for i in range(x_sample.shape[1]):
        x_sample[0,i] = x_sample[0,i] + delta #加delta
        y[i] = float(DNN_model.predict(x_sample))
        x_sample[0,i] = x_sample[0,i] - delta #恢复
        #print("取第{0}条数据，第{1}个变量上调{2}，y增加{3}".format(332, i, 0.01, y[i] - y_pred))
    sum_delt = sum(abs(y - y_pred))
    for i in range(x_sample.shape[1]):
        result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
        #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
    result = np.around(result, decimals=2)
    print(result)

#datasetX_bj[156,:]，[[ 0.3  -1.18 -0.29 -1.47  0.02  0.34  0.19 -0.36  0.19]]，从0.01往上加，至0.05有值，第1个变量贡献度100%
# Check(datasetX_bj,156,0.05)

#datasetX_bj[156,:]，[[ 0.21 -0.43 -0.56 -0.98  0.43  0.24  0.75  0.09  1.23]]，从0.01往上加，至0.05有值，第1个变量贡献度100%
# Check(datasetX_bj,157,0.05)

#datasetX_bj[156,:]，[[ 0.37 -1.17 -0.24 -0.48 -1.19  0.19  0.54 -0.59  1.26]]，从0.01往上加，至0.05有值，第1个变量贡献度100%
# Check(datasetX_bj,181,0.05)


#datasetX_bj[156,:]，[[ 0.25 -1.19  0.29 -1.33 -0.08  0.47  0.49 -0.37 -0.66]]，从0.01往上加，至0.0278有值，倒数第1个变量贡献度100%
# Check(datasetX_bj,297,0.0278)

#datasetX_hb[129,:]，[[ 0.32 -1.   -0.01 -0.8  -0.48 -0.36  0.57 -0.68  0.56]]，从0.01往上加，至0.017有值，倒数第1个变量贡献度100%
# Check(datasetX_hb,129,0.017)

#datasetX_hb[133,:]，[[-0.02 -0.13 -0.4  -0.37 -0.01 -0.28  0.16  0.01  0.93]]，从0.01往上加，至0.04有值，第1个变量贡献度100%
# Check(datasetX_hb,133,0.04)

#datasetX_hb[161,:]，[[ 0.18 -0.85 -0.07 -0.21 -0.45 -0.16 -0.04 -0.34  0.99]]，从0.01往上加，至0.05有值，第1个变量贡献度100%
# Check(datasetX_hb,161,0.05)
'''