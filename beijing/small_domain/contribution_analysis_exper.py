from tensorflow.keras.models import load_model
from sklearn import preprocessing
import numpy as np


np.set_printoptions(suppress=True)

file = np.load("D:/project/data/beijing/small_domain/dataset_abs_all.npy") #(365,42)
file[0,41] = -30 #根据平均值补全第一条记录
dataset_all = file #(365,42)

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

datasetX_all, datasetY_all = get_xy_dataset(dataset_all)
print(datasetX_all.shape,datasetY_all.shape) #datasetX_all(365,21) 顺序；datasetY_all(365,1) 顺序


#加载训练好的model
model = load_model("D:/project/data/beijing/small_domain/DNN_model.h5")

#contribution analysis
#Q:是否去归一化；只取一组x；上调ratio %；方法上：看y差异还是看mse差异？

"""
#实验一：未去归一化，只取一组x，多组ratio(结果：ratio取不同值时，一致性较好)
for j in np.arange(0.1,0.9,0.1):
    ratio = j
    x_sample = datasetX_all[0,:].reshape((-1,datasetX_all.shape[1])) #x_sample(1,21)
    y_pred = float(model.predict(x_sample)) #(1,1)
    y = np.zeros(x_sample.shape[1])
    for i in range(x_sample.shape[1]):
        x_sample[0,i] = x_sample[0,i] * (1 + ratio) #上调
        y[i] = float(model.predict(x_sample))
        x_sample[0,i] = x_sample[0,i] / (1 + ratio) #恢复
        print("第{0}个变量上调{1}%，y增加{2}".format(i, int(ratio*100), y[i] - y_pred))

    result = np.zeros(x_sample.shape[1])
    sum_delt = sum(abs(y - y_pred))
    for i in range(x_sample.shape[1]):
        result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
        #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
    print(result)
"""
"""
#实验二：未去归一化，取多组x，一组ratio(结果：x取不同值时，一致性很差，说明贡献度与X取值有关)
for j in range(5):
    print("取第%d组X时：" % j)
    ratio = 0.2
    print(datasetX_all[j,:])
    x_sample = datasetX_all[j,:].reshape((-1,datasetX_all.shape[1])) 
    y_pred = float(model.predict(x_sample)) #(1,1)
    y = np.zeros(x_sample.shape[1])
    for i in range(x_sample.shape[1]):
        x_sample[0,i] = x_sample[0,i] * (1 + ratio) #上调
        y[i] = float(model.predict(x_sample))
        x_sample[0,i] = x_sample[0,i] / (1 + ratio) #恢复
        #print("第{0}个变量上调{1}%，y增加{2}".format(i, int(ratio*100), y[i] - y_pred))

    result = np.zeros(x_sample.shape[1])
    sum_delt = sum(abs(y - y_pred))
    for i in range(x_sample.shape[1]):
        result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
        #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
    print(result)
"""
"""
#实验三：去归一化，只取一组x(结果：与未去归一化有一些差异。why？)
print("去归一化")
#ratio = 0.2
delta = 0.01
x_sample = datasetX_all[1,:].reshape((-1,datasetX_all.shape[1])) #x_sample(1,21)
y_pred = model.predict(x_sample) #(1,1)
y_pred = float(scaler2.inverse_transform(y_pred)) #去归一化
x_sample = scaler1.inverse_transform(x_sample) #去归一化
y = np.zeros(x_sample.shape[1])
for i in range(x_sample.shape[1]):
    x_sample[0,i] = x_sample[0,i] + delta #上调
    x_sample = scaler1.transform(x_sample) #归一化
    y[i] = model.predict(x_sample) #()
    y[i] = float(scaler2.inverse_transform(y[i].reshape((1,1))))
    x_sample = scaler1.inverse_transform(x_sample) #去归一化
    x_sample[0,i] = x_sample[0,i] - delta #恢复
    print("第{0}个变量加{1}，y增加{2}".format(i, delta, y[i] - y_pred))

result = np.zeros(x_sample.shape[1])
sum_delt = sum(abs(y - y_pred))
print("分母为：",sum_delt)
for i in range(x_sample.shape[1]):
    result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
    #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
result = np.around(result, decimals=2)
print(result)

print("未去归一化")
#ratio = 0.2
x_sample = datasetX_all[1,:].reshape((-1,datasetX_all.shape[1]))
y_pred = model.predict(x_sample) #(1,1)
y_pred = float(scaler2.inverse_transform(y_pred)) #去归一化
y = np.zeros(x_sample.shape[1])
for i in range(x_sample.shape[1]):
    x_sample[0,i] = x_sample[0,i] + delta
    y[i] = model.predict(x_sample) #()
    y[i] = float(scaler2.inverse_transform(y[i].reshape((1,1))))
    x_sample[0,i] = x_sample[0,i] - delta #恢复
    print("第{0}个变量上调{1}%，y增加{2}".format(i, delta, y[i] - y_pred))

result = np.zeros(x_sample.shape[1])
sum_delt = sum(abs(y - y_pred))
print("分母为：",sum_delt)
for i in range(x_sample.shape[1]):
    result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
    #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
result = np.around(result, decimals=2)
print(result)
"""
"""
#编为函数
def get_contribution_1(datasetX, x_num, NNmodel, inverse_transform = False, ratio = 0.2):
    x_sample = datasetX[x_num,:].reshape((-1,datasetX.shape[1])) #datasetX_all(16016,18) x_sample(1,18)
    y_pred = NNmodel.predict(x_sample) #(1,1)
    y = np.zeros(x_sample.shape[1])
    result = np.zeros(x_sample.shape[1])
    if inverse_transform is False:
        y_pred = float(NNmodel.predict(x_sample))
        for i in range(x_sample.shape[1]):
            x_sample[0,i] = x_sample[0,i] * (1 + ratio)
            y[i] = float(NNmodel.predict(x_sample))
            x_sample[0,i] = x_sample[0,i] / (1 + ratio) #恢复
            #print("第{0}个变量上调{1}%，y增加{2}".format(i, ratio*100, y[i] - y_pred))
    else:
        x_sample = scaler1.inverse_transform(x_sample)
        y_pred = float(scaler2.inverse_transform(y_pred))
        for i in range(x_sample.shape[1]):
            x_sample[0,i] = x_sample[0,i] * (1 + ratio)
            x_sample = scaler1.transform(x_sample)
            y[i] = NNmodel.predict(x_sample)
            y[i] = float(scaler2.inverse_transform(y[i].reshape((-1,1)))) ##去归一化
            x_sample = scaler1.inverse_transform(x_sample)
            x_sample[0,i] = x_sample[0,i] / (1 + ratio) #恢复
            #print("第{0}个变量上调{1}%，y增加{2}".format(i, ratio*100, y[i] - y_pred))
    sum_delt = sum(abs(y - y_pred))
    for i in range(x_sample.shape[1]):
        result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
        #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
    result = np.around(result, decimals=2)
    print(result)
    indices = np.argsort(result)[::-1] #从大到小的排序
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x_sample.shape[1]), result[indices], color="g", align="center")
    plt.xticks(range(x_sample.shape[1]), [indices[i]for i in indices], rotation='45')
    plt.xlim([-1, x_sample.shape[1]])
    plt.show()

#比较是否去归一化：
get_contribution_1(datasetX_all, 1, model, inverse_transform = False)
get_contribution_1(datasetX_all, 1, model, inverse_transform = True)

#比较选取的X
get_contribution_1(datasetX_all, 0, model, inverse_transform = True)
get_contribution_1(datasetX_all, 1, model, inverse_transform = True)
get_contribution_1(datasetX_all, 2, model, inverse_transform = True)
"""
"""
#datasetX_all[1,:] 0.001~0.1;datasetX_all[0,:] 0.001~0.04;datasetX_all[2,:] 0.001~0.01 选定：delta = 0.01
print("未去归一化")
for delta in np.arange(0.001,0.01,0.002):
    print("delta=",delta)
    x_sample = datasetX_all[34,:].reshape((-1,datasetX_all.shape[1]))
    y_pred = model.predict(x_sample) #(1,1)
    y_pred = float(scaler2.inverse_transform(y_pred)) #去归一化
    y = np.zeros(x_sample.shape[1])
    for i in range(x_sample.shape[1]):
        x_sample[0,i] = x_sample[0,i] + delta
        y[i] = model.predict(x_sample) #()
        y[i] = float(scaler2.inverse_transform(y[i].reshape((1,1))))
        x_sample[0,i] = x_sample[0,i] - delta #恢复
        #print("第{0}个变量上调{1}，y增加{2}".format(i, delta, y[i] - y_pred))

    result = np.zeros(x_sample.shape[1])
    sum_delt = sum(abs(y - y_pred))
    print("分母为：",sum_delt)
    for i in range(x_sample.shape[1]):
        result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
        #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
    result = np.around(result, decimals=2)
    print(result)
"""

"""
#另外一种思路：存储每一个特征缺失后所处测试的mse
from sklearn.metrics import mean_squared_error
import copy
X_train, y_train, X_test, y_test = split_dataset(datasetX_all, datasetY_all)
Mse = []
for i in range(18): 
    TmpTest = copy.copy(X_test)
    TmpTest[:,i] = 0
    y_pred = model.predict(TmpTest)
    TmpMse = mean_squared_error(y_test,y_pred)
    Mse.append(TmpMse)   
 
# 输出特征重要性索引， 误差从小到大，重要性由大到小
indices = np.argsort(Mse)  
for i in  range(18):
    print('特征排序')
    print ("{0} - {1:.3f}".format(indices[i], Mse[indices[i]]))
"""
"""
#实验：是否对y做去归一化处理（结论：结果一致）
print("对y去归一化")
delta = 0.01
x_sample = datasetX_all[0,:].reshape((-1,datasetX_all.shape[1])) #x_sample(1,21)
y_pred = model.predict(x_sample) #(1,1)
y_pred = float(scaler2.inverse_transform(y_pred)) #去归一化
y = np.zeros(x_sample.shape[1])
for i in range(x_sample.shape[1]):
    x_sample[0,i] = x_sample[0,i] + delta #上调
    y[i] = model.predict(x_sample) #()
    y[i] = float(scaler2.inverse_transform(y[i].reshape((1,1)))) #去归一化
    x_sample[0,i] = x_sample[0,i] - delta #恢复
    #print("第{0}个变量加{1}，y增加{2}".format(i, delta, y[i] - y_pred))
result = np.zeros(x_sample.shape[1])
sum_delt = sum(abs(y - y_pred))
for i in range(x_sample.shape[1]):
    result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
    #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
result = np.around(result, decimals=2)
print(result)

print("未对y去归一化")
x_sample = datasetX_all[0,:].reshape((-1,datasetX_all.shape[1]))
y_pred = float(model.predict(x_sample)) #(1,1)
y = np.zeros(x_sample.shape[1])
for i in range(x_sample.shape[1]):
    x_sample[0,i] = x_sample[0,i] + delta
    y[i] = float(model.predict(x_sample)) #()
    x_sample[0,i] = x_sample[0,i] - delta #恢复
    #print("第{0}个变量上调{1}，y增加{2}".format(i, delta, y[i] - y_pred))
result = np.zeros(x_sample.shape[1])
sum_delt = sum(abs(y - y_pred))
for i in range(x_sample.shape[1]):
    result[i] = (abs(y[i] - y_pred) / sum_delt) * 100
    #print("第{0}个变量贡献度为{1}%".format(i, result[i]))
result = np.around(result, decimals=2)
print(result)
"""