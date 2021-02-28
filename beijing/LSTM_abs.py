import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import math
from sklearn import metrics

dataset = np.load("E:/project/beijing/dataset_abs.npy")
#np.random.shuffle(dataset) #(365,25)

var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'RH_Bias':13, 'TEM_Bias':14, 'WSPD_Bias':15, 'WDIR_Bias':16, 'PRE_Bias':17, 'RH_Obs':18, 'TEM_Obs':19, 'WSPD_Obs':20, 'WDIR_Obs':21, 'PRE_Obs':22, 'PBLH_Sim':23, 'SOLRAD_Sim':24}

# 数据集制作
var_sele = ['PM2.5_Sim','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WSPD_Bias','WDIR_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']
datasetY = dataset[:,0]
datasetX = np.zeros((365,len(var_sele))) #(365,4)
i = 0
for var in var_sele:
    datasetX[:,i] = dataset[:,var_dict.get(var)]
    i += 1
#归一化
scaler1 = preprocessing.StandardScaler().fit(datasetX)
datasetX = scaler1.transform(datasetX)

datasetY = datasetY.reshape((datasetY.shape[0],1)) #(365,1)
scaler2 = preprocessing.StandardScaler().fit(datasetY)
datasetY = scaler2.transform(datasetY)
# 训练集、测试集
ratio = 0.8
train_size=int(len(datasetX)*ratio) 
X_train,y_train=datasetX[0:train_size,:],datasetY[0:train_size] 
X_test,y_test=datasetX[train_size:len(datasetX),:],datasetY[train_size:len(datasetX)]
X_train=X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
X_test=X_test.reshape((X_test.shape[0],1,X_test.shape[1]))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#(292, 1, 4) (292, 1) (73, 1, 4) (73, 1)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=20, 
    restore_best_weights=True)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001, 
    decay_steps=35, 
    decay_rate=1,
    staircase=False)

def build_model():
    model = keras.Sequential([
    layers.LSTM(36, activation='relu', return_sequences = True, input_shape= (X_train.shape[1],X_train.shape[2])),
    layers.Dropout(0.5),
    layers.Dense(1)])
    optimizer=tf.keras.optimizers.Adam(lr_schedule)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

model = build_model()
model.summary()

EPOCHS = 1000
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=5, validation_split = 0.2, callbacks=[early_stopping], shuffle=False)

#查看loss
loss = history.history['loss'] #mse
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']

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


y_pred = model.predict(X_test) #(73,1,1)
y_pred = y_pred.reshape((len(X_test),1)) #(73,1)
y_pred = scaler2.inverse_transform(y_pred)
test_rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Test RMSE: %.3f' % test_rmse)

#bias=sim-obs,PM25_revised=sim-bias
datasetX = datasetX.reshape((datasetX.shape[0], 1, datasetX.shape[1]))
bias_pred = model.predict(datasetX) #(365, 1, 1)
bias_pred = bias_pred.reshape((len(bias_pred),1))
bias_pred = scaler2.inverse_transform(bias_pred) #(365,1)
PM25_revised = dataset[:,var_dict.get('PM2.5_Sim')] - bias_pred.reshape(365,)

RMSE_revised = math.sqrt(metrics.mean_squared_error(dataset[:,var_dict.get('PM2.5_Obs')], PM25_revised))
v = np.concatenate((dataset[:,var_dict.get('PM2.5_Obs')], PM25_revised)).reshape((2,365))
R2_revised = np.corrcoef(v)[0,1] ** 2
NMB_revised = np.sum(PM25_revised - dataset[:,var_dict.get('PM2.5_Obs')]) / np.sum(dataset[:,var_dict.get('PM2.5_Obs')])

RMSE_original = math.sqrt(metrics.mean_squared_error(dataset[:,var_dict.get('PM2.5_Obs')], dataset[:,var_dict.get('PM2.5_Sim')]))
u = np.concatenate((dataset[:,var_dict.get('PM2.5_Obs')], dataset[:,var_dict.get('PM2.5_Sim')])).reshape((2,365))
R2_original = np.corrcoef(u)[0,1] ** 2
NMB_original = np.sum(dataset[:,var_dict.get('PM2.5_Sim')] - dataset[:,var_dict.get('PM2.5_Obs')]) / np.sum(dataset[:,var_dict.get('PM2.5_Obs')])

print("final RMSE: %.3f" % RMSE_revised, " ,original RMSE: %.3f" % RMSE_original)
print("final R2: %.3f" % R2_revised, " ,original R2: %.3f" % R2_original)
print("final NMB: %.3f" % NMB_revised, " ,original NMB: %.3f" % NMB_original)




