import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import math
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler

np.random.seed(42)
file = np.load("E:/project/BTH/dataset_abs.npy") #(44,364,23)
dataset_winter = np.concatenate((file[:,:60,:],file[:,334:,:]),axis=1).reshape((-1,23)) #(3960, 23)
dataset_all = file.reshape((-1,23)) #(16016, 23)
var_dict = {'PM2.5_Bias':0, 'NO2_Bias':1, 'SO2_Bias':2, 'O3_Bias':3, 'CO_Bias':4, 'PM2.5_Obs':5, 'NO2_Obs':6, 'SO2_Obs':7, 'O3_Obs':8, 'CO_Obs':9, 'PM2.5_Sim':10, 'RH_Bias':11, 'TEM_Bias':12, 'WSPD_Bias':13, 'WDIR_Bias':14, 'PRE_Bias':15, 'RH_Obs':16, 'TEM_Obs':17, 'WSPD_Obs':18, 'WDIR_Obs':19, 'PRE_Obs':20, 'PBLH_Sim':21, 'SOLRAD_Sim':22}
np.random.shuffle(dataset_all)
# 数据集制作
var_sele = ['PM2.5_Sim','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WSPD_Bias','WDIR_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']

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
# 训练集、测试集
def split_dataset(input_datasetx, input_datasety, ratio = 0.75):
    train_size = int(len(input_datasetx)*ratio) 
    X_train, y_train = input_datasetx[0:train_size,:], input_datasety[0:train_size] 
    X_test, y_test = input_datasetx[train_size:,:], input_datasety[train_size:]
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = split_dataset(datasetX_all, datasetY_all)

# 输出每个特征的贡献度
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

#建立模型
dt = DecisionTreeRegressor(criterion='mse',max_depth=14) 
dt.fit(X_train, y_train) 
y_hat = dt.predict(X_test) 

# 输出预测得分
print(dt.score(X_train, y_train))
print(dt.score(X_test,y_test))


#特征重要度
features = list(X_test[1,:])
importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
num_features = len(importances)
     
    #将特征重要度以柱状图展示
plt.figure()
plt.title("Feature importances")
plt.bar(range(num_features), importances[indices], color="g", align="center")
plt.xticks(range(num_features), [indices[i]for i in indices], rotation='45')
plt.xlim([-1, num_features])
plt.show()
  #输出各个特征的重要度
for i in indices:
    print('特征排序')
    print ("{0} - {1:.3f}".format(indices[i], importances[i]))