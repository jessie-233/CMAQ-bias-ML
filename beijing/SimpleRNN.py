import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


np.random.seed(42)
dataset = np.load("E:/project/beijing/dataset.npy")
np.random.shuffle(dataset) #(365,25)

var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'RH_Bias':13, 'TEM_Bias':14, 'WSPD_Bias':15, 'WDIR_Bias':16, 'PRE_Bias':17, 'RH_Obs':18, 'TEM_Obs':19, 'WSPD_Obs':20, 'WDIR_Obs':21, 'PRE_Obs':22, 'PBLH_Sim':23, 'SOLRAD_Sim':24}

# 数据集制作
var_sele = ['NO2_Bias', 'SO2_Bias', 'NO2_Obs', 'SO2_Obs']
datasetY = dataset[:,0]
datasetX = np.zeros((365,len(var_sele)))
i = 0
for var in var_sele:
    datasetX[:,i] = dataset[:,var_dict.get(var)]
    i += 1
#归一化
datasetX = preprocessing.scale(datasetX)

# 训练集、测试集
ratio = 0.9
train_size=int(len(datasetX)*ratio) #328
X_train,y_train=datasetX[0:train_size,:],datasetY[0:train_size] #0:328
X_test,y_test=datasetX[train_size:len(datasetX),:],datasetY[train_size:len(datasetX)] #328:365

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001, 
    decay_steps=5, 
    decay_rate=1,
    staircase=False)

def build_model():
    model = keras.Sequential([
    layers.SimpleRNN(32, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(1)])
    optimizer = tf.keras.optimizers.RMSprop(lr_schedule)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

model = build_model()
model.summary()

EPOCHS = 100
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split = 0.2, callbacks=[early_stopping])

#查看loss(mse)
loss = history.history['loss']
val_loss = history.history['val_loss']
#mae = history.history['mae']
#val_mae = history.history['val_mae']

epochs_range = range(100)
plt.figure(figsize=(8, 8))
plt.plot(epochs_range, loss, label='Training Error')
plt.plot(epochs_range, val_loss, label='Validation Error')
plt.legend(loc='upper right')
plt.title('Training and Validation Error')
plt.show()




