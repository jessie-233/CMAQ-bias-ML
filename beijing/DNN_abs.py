import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import math
from sklearn import metrics


np.random.seed(42)
file = np.load("E:/project/beijing/dataset_abs.npy") #(365,25)
dataset_all = file #(365,25)
dataset_winter = np.concatenate((file[:60,:],file[334:,:]),axis=0) #(91,25)
var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'RH_Bias':13, 'TEM_Bias':14, 'WSPD_Bias':15, 'WDIR_Bias':16, 'PRE_Bias':17, 'RH_Obs':18, 'TEM_Obs':19, 'WSPD_Obs':20, 'WDIR_Obs':21, 'PRE_Obs':22, 'PBLH_Sim':23, 'SOLRAD_Sim':24}
np.random.shuffle(dataset_all) #(365,25)
# 数据集制作
var_sele = ['PM2.5_Sim','NO2_Bias','SO2_Bias','O3_Bias','CO_Bias','NO2_Obs','SO2_Obs','O3_Obs','CO_Obs','RH_Bias','TEM_Bias','WSPD_Bias','WDIR_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim'] #缺少WDIR_Obs (one-hot)、PM10_Bias、PM10_Obs、PM2.5_Obs

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
def split_dataset(input_datasetx, input_datasety, ratio = 0.8):
    train_size = int(len(input_datasetx)*ratio) 
    X_train, y_train = input_datasetx[0:train_size,:], input_datasety[0:train_size] 
    X_test, y_test = input_datasetx[train_size:,:], input_datasety[train_size:]
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = split_dataset(datasetX_all, datasetY_all)
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#(292, 20) (292, 1) (73, 20) (73, 1)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=50, 
    restore_best_weights=True)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001, 
    decay_steps=24*100, 
    decay_rate=1,
    staircase=False)


def build_model():
    model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(8, activation='relu'),
    #layers.Dense(4, activation='relu'),
    #layers.Dense(3, activation='relu'),
    layers.Dense(1, activation='linear')])
    optimizer=tf.keras.optimizers.RMSprop(lr_schedule)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

model = build_model()
model.summary()

EPOCHS = 1000
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=10, validation_split = 0.2, callbacks=[early_stopping], shuffle=False)

#查看loss
loss = history.history['loss'] #mse
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']

#画图
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, loss, label='Training MSE')
plt.plot(history.epoch, val_loss, label='Validation MSE')
plt.legend(loc='upper right')
plt.title('Training and Validation MSE')

plt.subplot(1, 2, 2)
plt.plot(history.epoch, mae, label='Training MAE')
plt.plot(history.epoch, val_mae, label='Validation MAE')
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

model.save("E:/project/beijing/DNN_abs_model.h5")









