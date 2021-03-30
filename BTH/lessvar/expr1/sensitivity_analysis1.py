from tensorflow.keras.models import load_model
from sklearn import preprocessing
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True) #抑制使用对小数的科学记数法
np.set_printoptions(precision=2) #设置输出的精度
#创建excel，北京贡献度'bj-detail'，'bj-overall'，天津、河北 共6个sheets
data = np.load("D:/project/data/BTH/dataset_abs_corr2.npy") #(44, 363, 42)
all_vars = {'PM2.5_Bias', 'PM10_Bias', 'NO2_Bias', 'SO2_Bias', 'O3_Bias', 'CO_Bias', 'PM2.5_Obs', 'PM10_Obs', 'NO2_Obs', 'SO2_Obs', 'O3_Obs', 'CO_Obs', 'PM2.5_Sim','PM10_Sim','NO2_Sim','SO2_Sim','O3_Sim','CO_Sim', 'RH_Bias', 'TEM_Bias', 'WSPD_Bias', 'WDIR_Bias', 'PRE_Bias', 'RH_Obs', 'TEM_Obs', 'WSPD_Obs', 'WDIR_Obs', 'PRE_Obs', 'PBLH_Sim', 'SOLRAD_Sim','RH_Sim','TEM_Sim','WSPD_Sim','WDIR_Sim','PRE_Sim', 'WIN_N_Obs','WIN_N_Sim', 'WIN_E_Obs','WIN_E_Sim', 'WIN_N_Bias', 'WIN_E_Bias', 'PM2.5_Bias_ystd'}
var_dict = {'PM2.5_Bias':0, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'PM2.5_Obs':6, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'PM2.5_Sim':12, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'WIN_N_Obs':35, 'WIN_E_Obs':37, 'WIN_N_Bias':39, 'WIN_E_Bias':40, 'PM2.5_Bias_ystd':41}
# var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WDIR_Bias','WSPD_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']
var_sele = ['PM2.5_Bias_ystd','PM2.5_Sim','NO2_Bias','RH_Bias','O3_Bias','SO2_Bias','WSPD_Bias','NO2_Obs','O3_Obs']

region_num = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,2,2,0,0,0,2,0,0,2,2,2,2]) #北京0，天津1，河北2
bj_data = np.mean(data[np.where(region_num == 0),:,:],axis=(0,1)) #(363, 42)
tj_data = np.mean(data[np.where(region_num == 1),:,:],axis=(0,1))
hb_data = np.mean(data[np.where(region_num == 2),:,:],axis=(0,1))

#or不取平均
# bj_data = data[np.where(region_num == 0),:,:].reshape((-1,len(all_vars))) #(2904,42)
# tj_data = data[np.where(region_num == 1),:,:].reshape((-1,len(all_vars)))
# hb_data = data[np.where(region_num == 2),:,:].reshape((-1,len(all_vars)))

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
DNN_model = load_model("D:/project/data/BTH/lessvar/expr1/DNN_model1.h5")
'''
#制作sheets
#bj_detail
bj_detail = np.zeros((datasetX_bj.shape[0],datasetX_bj.shape[1]))  
for i in range(datasetX_bj.shape[0]):
    print(i)
    bj_detail[i,:] = get_contribution(datasetX_bj, i, DNN_model)
bj_detail_pd = pd.DataFrame(bj_detail,index=np.arange(1,len(datasetX_bj)+1),columns=var_sele)
#tj_detail
tj_detail = np.zeros((datasetX_tj.shape[0],datasetX_tj.shape[1]))  
for i in range(datasetX_tj.shape[0]):
    print(i)
    tj_detail[i,:] = get_contribution(datasetX_tj, i, DNN_model)
tj_detail_pd = pd.DataFrame(tj_detail,index=np.arange(1,len(datasetX_tj)+1),columns=var_sele)
#hb_detail
hb_detail = np.zeros((datasetX_hb.shape[0],datasetX_hb.shape[1]))  
for i in range(datasetX_hb.shape[0]):
    print(i)
    hb_detail[i,:] = get_contribution(datasetX_hb, i, DNN_model)
hb_detail_pd = pd.DataFrame(hb_detail,index=np.arange(1,len(datasetX_hb)+1),columns=var_sele)
# #bj_overall
# bj_overall = pd.DataFrame({'PM2.5_Sim':bj_detail[:,0],
# 'PM2.5_Bias_ystd':bj_detail[:,1],
# 'emmision_bias':bj_detail_pd['NO2_Bias']+bj_detail_pd['SO2_Bias']+bj_detail_pd['O3_Bias'],
# 'emmision_base':bj_detail_pd['NO2_Obs']+bj_detail_pd['SO2_Obs']+bj_detail_pd['O3_Obs'],
# 'mete_bias':bj_detail_pd['RH_Bias']+bj_detail_pd['TEM_Bias']+bj_detail_pd['WDIR_Bias']+bj_detail_pd['WSPD_Bias']+bj_detail_pd['PRE_Bias'],
# 'mete_base':bj_detail_pd['RH_Obs']+bj_detail_pd['TEM_Obs']+bj_detail_pd['WSPD_Obs']+bj_detail_pd['PRE_Obs']+bj_detail_pd['PBLH_Sim']+bj_detail_pd['SOLRAD_Sim']},
# index=bj_detail_pd.index)
# #tj_overall
# tj_overall = pd.DataFrame({'PM2.5_Sim':tj_detail[:,0],
# 'PM2.5_Bias_ystd':tj_detail[:,1],
# 'emmision_bias':tj_detail_pd['NO2_Bias']+tj_detail_pd['SO2_Bias']+tj_detail_pd['O3_Bias'],
# 'emmision_base':tj_detail_pd['NO2_Obs']+tj_detail_pd['SO2_Obs']+tj_detail_pd['O3_Obs'],
# 'mete_bias':tj_detail_pd['RH_Bias']+tj_detail_pd['TEM_Bias']+tj_detail_pd['WDIR_Bias']+tj_detail_pd['WSPD_Bias']+tj_detail_pd['PRE_Bias'],
# 'mete_base':tj_detail_pd['RH_Obs']+tj_detail_pd['TEM_Obs']+tj_detail_pd['WSPD_Obs']+tj_detail_pd['PRE_Obs']+tj_detail_pd['PBLH_Sim']+tj_detail_pd['SOLRAD_Sim']},
# index=tj_detail_pd.index)
# #hb_overall
# hb_overall = pd.DataFrame({'PM2.5_Sim':hb_detail[:,0],
# 'PM2.5_Bias_ystd':hb_detail[:,1],
# 'emmision_bias':hb_detail_pd['NO2_Bias']+hb_detail_pd['SO2_Bias']+hb_detail_pd['O3_Bias'],
# 'emmision_base':hb_detail_pd['NO2_Obs']+hb_detail_pd['SO2_Obs']+hb_detail_pd['O3_Obs'],
# 'mete_bias':hb_detail_pd['RH_Bias']+hb_detail_pd['TEM_Bias']+hb_detail_pd['WDIR_Bias']+hb_detail_pd['WSPD_Bias']+hb_detail_pd['PRE_Bias'],
# 'mete_base':hb_detail_pd['RH_Obs']+hb_detail_pd['TEM_Obs']+hb_detail_pd['WSPD_Obs']+hb_detail_pd['PRE_Obs']+hb_detail_pd['PBLH_Sim']+hb_detail_pd['SOLRAD_Sim']},
# index=hb_detail_pd.index)
#写入excel
writer = pd.ExcelWriter("D:/project/data/BTH/lessvar/expr1/sensitivity_analysis_temp.xlsx")
bj_detail_pd.to_excel(writer,'bj_detail',float_format='%.1f')
tj_detail_pd.to_excel(writer,'tj_detail',float_format='%.1f')
hb_detail_pd.to_excel(writer,'hb_detail',float_format='%.1f')

# bj_overall.to_excel(writer,'bj_overall',float_format='%.1f')
# tj_overall.to_excel(writer,'tj_overall',float_format='%.1f')
# hb_overall.to_excel(writer,'hb_overall',float_format='%.1f')
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
