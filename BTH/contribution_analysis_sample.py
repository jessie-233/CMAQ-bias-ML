# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 22:48:21 2021

@author: Evans
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn
import keras
from keras import layers
import tensorflow as tf
import math
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import LearningRateScheduler

np.random.seed(42)
file = np.load("dataset_abs.npy") #(44,364,23)
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

def build_model():
    # 构建神经网络模型
    model = Sequential()
    model.add(Dense(16, input_dim=18, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',optimizer='rmsprop')
    return model

model = build_model()
#model.summary()
print(model.summary())   # 显示构建模型结构


# 定义并拟合模型
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)
history = model.fit( X_train, y_train, batch_size=64, epochs=300, verbose=1, callbacks=None,
validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
#查看loss
loss = history.history['loss'] #mse
val_loss = history.history['val_loss']
#mae = history.history['mae']
#val_mae = history.history['val_mae']

#画图
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, loss, label='Training MSE')
plt.plot(history.epoch, val_loss, label='Validation MSE')
plt.legend(loc='upper right')
plt.title('Training and Validation MSE')

plt.subplot(1, 2, 2)
plt.plot(history.epoch, loss, label='Training MAE')
plt.plot(history.epoch, val_loss, label='Validation MAE')
plt.legend(loc='upper right')
plt.title('Training and Validation MAE')
plt.show()

#计算测试集RMSE损失
y_pred = model.predict(X_test)
y_pred = scaler2.inverse_transform(y_pred)
y_test = scaler2.inverse_transform(y_test)
test_rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('测试集RMSE损失为：%.3f' % test_rmse)

#计算修正后的各项指标
#bias=sim-obs,PM25_revised=sim-bias
#RMSE
def cal_rmse(xtest, ytest = None, info = ''):
    if ytest is None:
        RMSE = math.sqrt(metrics.mean_squared_error(xtest[:,var_dict.get('PM2.5_Obs')], xtest[:,var_dict.get('PM2.5_Sim')]))
    else:
        y_pred = model.predict(xtest)
        y_pred = scaler2.inverse_transform(y_pred)
        PM25_revised = ytest[:,var_dict.get('PM2.5_Sim')] - y_pred.reshape(len(xtest),)
        RMSE = math.sqrt(metrics.mean_squared_error(ytest[:,var_dict.get('PM2.5_Obs')], PM25_revised))
    print(info + ' RMSE: %.3f' % RMSE)

cal_rmse(dataset_all, info = 'all year')
cal_rmse(datasetX_all, dataset_all, 'all year revised')
cal_rmse(dataset_winter, info = 'winter')
cal_rmse(datasetX_winter, dataset_winter, 'winter revised')

#R2
def cal_R2(xtest, ytest = None, info = ''):
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

cal_R2(dataset_all, info = 'all year')
cal_R2(datasetX_all, dataset_all, 'all year revised')
cal_R2(dataset_winter, info = 'winter')
cal_R2(datasetX_winter, dataset_winter, 'winter revised')

#NMB
def cal_NMB(xtest, ytest = None, info = ''):
    if ytest is None:
        NMB = np.sum(xtest[:,var_dict.get('PM2.5_Sim')] - xtest[:,var_dict.get('PM2.5_Obs')]) / np.sum(xtest[:,var_dict.get('PM2.5_Obs')])
    else:
        y_pred = model.predict(xtest)
        y_pred = scaler2.inverse_transform(y_pred)
        #bias=sim-obs,PM25_revised=sim-bias
        PM25_revised = ytest[:,var_dict.get('PM2.5_Sim')] - y_pred.reshape(len(xtest),)
        NMB = np.sum(PM25_revised - ytest[:,var_dict.get('PM2.5_Obs')]) / np.sum(ytest[:,var_dict.get('PM2.5_Obs')])
    print(info + ' NMB: %.3f' % NMB)

cal_NMB(dataset_all, info = 'all year')
cal_NMB(datasetX_all, dataset_all, 'all year revised')
cal_NMB(dataset_winter, info = 'winter')
cal_NMB(datasetX_winter, dataset_winter, 'winter revised')

# 输出每个特征的贡献度
# 存储每一个特征缺失后所处测试的mse
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
