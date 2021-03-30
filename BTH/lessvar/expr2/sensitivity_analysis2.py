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
DNN_model = load_model("D:/project/data/BTH/lessvar/expr2/DNN_model2.h5")
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
writer = pd.ExcelWriter("D:/project/data/BTH/lessvar/expr2/sensitivity_analysis_temp.xlsx")
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
    # print(get_contribution(datasetX, day_num, DNN_model)) #confirm the problem
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

#datasetX_bj[156,:]，[[ 0.96  2.67 -0.1   0.75  0.94  0.13 -0.68  0.78 -0.94]]，从0.01往上加，至0.8有值，倒数第2个变量贡献度100%
# Check(datasetX_tj,64,0.8)

#datasetX_bj[156,:]，[[ 1.69  1.21  1.06  0.5   0.52  0.93  0.69 -0.96  0.96]]，从0.01往上加，至0.5有值，第2个变量贡献度100%
# Check(datasetX_tj,189,0.5)

#datasetX_bj[156,:]，[[ 0.98  1.08  1.47  0.1   0.35  0.93 -0.37 -1.1   1.12]]，从0.01往上加，至0.55有值，第2个变量贡献度100%
# Check(datasetX_tj,190,0.55)


#datasetX_bj[156,:]，[[ 1.85  2.25  0.49 -0.3  -0.87  0.85  0.08  1.    0.87]]，从0.01往上加，至1.5有值，倒数第2个变量贡献度100%
# Check(datasetX_tj,286,1.5)

#datasetX_hb[129,:]，[[ 1.44  1.86 -0.09  1.14  1.29  0.59 -0.5   0.27 -0.79]]，从0.01往上加，至0.017有值，倒数第1个变量贡献度100%
# Check(datasetX_hb,129,0.017)

#datasetX_hb[133,:]，[[ 1.44  1.86 -0.09  1.14  1.29  0.59 -0.5   0.27 -0.79]]，从0.01往上加，至2有值，倒数第2个变量贡献度100%
# Check(datasetX_tj,290,2)

#datasetX_hb[161,:]，[[ 1.1   4.22  2.04  1.19 -0.41  2.05  0.03  1.06 -1.3 ]]，从0.01往上加，至3有值，倒数第2个变量贡献度100%
# Check(datasetX_tj,314,3)

#datasetX_hb[161,:]，[[ 2.84  1.96  0.29  0.78  0.58  0.27 -0.21  1.08 -0.96]]，从0.01往上加，至2有值，倒数第2个变量贡献度100%
# Check(datasetX_tj,315,2)

#datasetX_hb[161,:]，[[ 0.84  1.52  0.99 -0.12  0.32  1.91 -0.3   0.12 -1.4 ]]，从0.01往上加，至0.72有值，倒数第2个变量贡献度100%
# Check(datasetX_tj,338,0.72)


#datasetX_hb[161,:]，[[ 1.77  3.08  1.41 -0.12 -0.04  2.09 -0.41  0.77 -1.47]]，从0.01往上加，至2有值，倒数第2个变量贡献度100%
# Check(datasetX_tj,339,2)

#datasetX_hb[161,:]，[[ 1.06  3.45  0.94  0.6   0.02  1.85 -0.29  1.02 -1.47]]，从0.01往上加，至2有值，倒数第2个变量贡献度100%
# Check(datasetX_tj,340,2)

#datasetX_hb[161,:]，[[ 1.56  1.87 -0.85  0.18  0.95  0.03  0.21  0.72 -0.4 ]]，从0.01往上加，至1.3有值，倒数第2个变量贡献度100%
# Check(datasetX_hb,290,1.3)


