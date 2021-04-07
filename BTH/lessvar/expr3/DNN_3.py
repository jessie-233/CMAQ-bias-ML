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

np.random.seed(982)
tf.random.set_seed(571)

data = np.load("D:/project/data/BTH/DOMAIN_TRANS/BTH/dataset_BTH.npy") #(44,363,43)
dataset_all = data.reshape((-1,43))
train_data = np.concatenate((data[:,0:20,:],data[:,29:49,:],data[:,57:77,:],data[:,88:108,:],data[:,118:138,:],data[:,149:169,:],data[:,179:199,:],data[:,210:230,:],data[:,241:261,:],data[:,271:291,:],data[:,302:322,:],data[:,332:352,:]),axis=1).reshape((-1,43)) #(10560, 43)
test_data = np.concatenate((data[:,20:29,:],data[:,49:57,:],data[:,77:88,:],data[:,108:118,:],data[:,138:149,:],data[:,169:179,:],data[:,199:210,:],data[:,230:241,:],data[:,261:271,:],data[:,291:302,:],data[:,322:332,:],data[:,352:363,:]),axis=1).reshape((-1,43)) #(5412, 43)

var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'PM10_Sim':13, 'NO2_Sim':14, 'SO2_Sim':15, 'O3_Sim':16, 'CO_Sim':17, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'RH_Sim':30, 'TEM_Sim':31, 'WSPD_Sim':32, 'WDIR_Sim':33, 'PRE_Sim':34, 'PM2.5_Bias_ystd':35, 'NO2_Bias_ystd':36, 'RH_Bias_ystd':37, 'O3_Bias_ystd':38, 'SO2_Bias_ystd':39, 'WSPD_Bias_ystd':40, 'NO2_Obs_ystd':41, 'O3_Obs_ystd':42}

np.random.shuffle(train_data) #(16016, 43)
var_sele = ['PM2.5_Bias_ystd','PM2.5_Sim','NO2_Bias','RH_Bias','SO2_Bias','WSPD_Bias','NO2_Obs','TEM_Obs','RH_Obs']

#standardization: scaler of dataset_all
Y_all = dataset_all[:,0].reshape((dataset_all.shape[0],1))
X_all = np.zeros((len(dataset_all),len(var_sele)))
i = 0
for var in var_sele:
    X_all[:,i] = dataset_all[:,var_dict.get(var)]
    i += 1
scaler1 = preprocessing.StandardScaler().fit(X_all)
scaler2 = preprocessing.StandardScaler().fit(Y_all)
X_all = scaler1.transform(X_all)
Y_all = scaler2.transform(Y_all) #(15972, 1)

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

#train dataset & test dataset
X_train , y_train = get_xy_dataset(train_data)
X_test, y_test = get_xy_dataset(test_data)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=20, 
    restore_best_weights=True)
    
#initial_learning_rate / (1 + decay_rate * step / decay_step)
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001, 
    decay_steps=160*100, 
    decay_rate=1,
    staircase=False)

# #option: LearningRateScheduler
# def scheduler(epoch):
#     # 每隔40个epoch，学习率减小为原来的1/10
#     if epoch % 40 == 0 and epoch != 0:
#         lr = K.get_value(model.optimizer.lr)
#         K.set_value(model.optimizer.lr, lr * 0.1)
#         print("lr changed to {}".format(lr * 0.1))
#     return K.get_value(model.optimizer.lr)
# reduce_lr = LearningRateScheduler(scheduler)

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
#loss
loss = history.history['loss'] #mse
val_loss = history.history['val_loss']
# mae = history.history['mae']
# val_mae = history.history['val_mae']

#plot
plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
plt.plot(history.epoch, loss, label='Training MSE')
plt.plot(history.epoch, val_loss, label='Validation MSE')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss(MSE)')
plt.savefig("D:/project/data/BTH/lessvar/expr3/DNN_Model_Loss3.png")
# plt.subplot(1, 2, 2)
# plt.plot(history.epoch, mae, label='Training MAE')
# plt.plot(history.epoch, val_mae, label='Validation MAE')
# plt.legend(loc='upper right')
# plt.title('Training and Validation MAE')
plt.show()

#RMSE on test dataset
y_pred = model.predict(X_test)
y_pred = scaler2.inverse_transform(y_pred) #invers
y_test = scaler2.inverse_transform(y_test) #invers
test_rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('RMSE on test dataset：%.3f' % test_rmse)


#RMSE
def cal_rmse(xtest, ytest = None, info = ''):
    if ytest is None:
        RMSE = math.sqrt(metrics.mean_squared_error(xtest[:,var_dict.get('PM2.5_Obs')], xtest[:,var_dict.get('PM2.5_Sim')]))
    else:
        y_pred = model.predict(xtest)
        y_pred = scaler2.inverse_transform(y_pred) #invers
        PM25_revised = ytest[:,var_dict.get('PM2.5_Sim')] - y_pred.reshape(len(xtest),)
        RMSE = math.sqrt(metrics.mean_squared_error(ytest[:,var_dict.get('PM2.5_Obs')], PM25_revised))
    print(info + ' RMSE: %.3f' % RMSE)

cal_rmse(dataset_all, info = 'all year')
cal_rmse(X_all, dataset_all, 'all year revised')

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
cal_R2(X_all, dataset_all, 'all year revised')

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
cal_NMB(X_all, dataset_all, 'all year revised')


model.save("D:/project/data/BTH/lessvar/expr3/DNN_model3.h5")

