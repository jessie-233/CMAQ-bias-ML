import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
import math
from sklearn import metrics

np.random.seed(876)
tf.random.set_seed(925)
file = np.load("D:/project/data/BTH/new2/dataset_abs_corr2.npy") #(44,363,42)
dataset_winter = np.concatenate((file[:,:60,:],file[:,334:,:]),axis=1).reshape((-1,42)) #(3960, 42)
dataset_all = file.reshape((-1,42)) #(16016, 42)
var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'WIN_N_Obs':35, 'WIN_E_Obs':37, 'WIN_N_Bias':39, 'WIN_E_Bias':40, 'PM2.5_Bias_ystd':41}
np.random.shuffle(dataset_all) #(16016, 42)
# 数据集制作
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WDIR_Bias','WSPD_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']

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
datasetX_all, datasetY_all = get_xy_dataset(dataset_all) #scaler1, scaler2目前是全集数据的
# 训练集、测试集
def split_dataset(input_datasetx, input_datasety, ratio = 0.75):
    train_size = int(len(input_datasetx)*ratio) 
    X_train, y_train = input_datasetx[0:train_size,:], input_datasety[0:train_size] 
    X_test, y_test = input_datasetx[train_size:,:], input_datasety[train_size:]
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = split_dataset(datasetX_all, datasetY_all)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=50, 
    restore_best_weights=True)
    
#initial_learning_rate / (1 + decay_rate * step / decay_step)
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001, 
    decay_steps=160*100, 
    decay_rate=1,
    staircase=False)

#另外一种学习率下降方法
def scheduler(epoch):
    # 每隔40个epoch，学习率减小为原来的1/10
    if epoch % 40 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)

def build_model():
    model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(4, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(3, activation='relu'),
    layers.Dense(1, activation='linear')])
    optimizer=tf.keras.optimizers.RMSprop(lr_schedule)
    model.compile(loss='mse', optimizer=optimizer)
    return model

model = build_model()
model.summary()

EPOCHS = 1000
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=60, validation_split = 0.2, callbacks=[early_stopping], shuffle=False)
#查看loss
loss = history.history['loss'] #mse
val_loss = history.history['val_loss']
# mae = history.history['mae']
# val_mae = history.history['val_mae']

#画图
plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
plt.plot(history.epoch, loss, label='Training MSE')
plt.plot(history.epoch, val_loss, label='Validation MSE')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss(MSE)')
plt.savefig("D:/project/data/BTH/new/DNN_Model_Loss.png")
# plt.subplot(1, 2, 2)
# plt.plot(history.epoch, mae, label='Training MAE')
# plt.plot(history.epoch, val_mae, label='Validation MAE')
# plt.legend(loc='upper right')
# plt.title('Training and Validation MAE')
plt.show()

#计算测试集RMSE损失
y_pred = model.predict(X_test)
y_pred = scaler2.inverse_transform(y_pred) #用全集的scaler
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
        y_pred = scaler2.inverse_transform(y_pred) #计算winter时用全集的scaler
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

model.save("D:/project/data/BTH/new2/DNN_model.h5")