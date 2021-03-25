import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'
pd.set_option('precision',2)#输出精度2

'''
var_dict = {'PM2.5_Bias':0, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'PM2.5_Obs':6, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'PM2.5_Sim':12, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'WIN_N_Obs':35, 'WIN_E_Obs':37, 'WIN_N_Bias':39, 'WIN_E_Bias':40, 'PM2.5_Bias_ystd':41}
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WDIR_Bias','WSPD_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']
data = np.load("D:/project/data/BTH/new/dataset_abs_corr2.npy") #(44, 363, 42)
region_num = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,2,2,0,0,0,2,0,0,2,2,2,2]) #北京0，天津1，河北2
a = data[np.where(region_num == 0),:,:] #(1, 8, 363, 42)
#取平均
bj_data = np.mean(data[np.where(region_num == 0),:,:],axis=(0,1)) #(363, 42)
tj_data = np.mean(data[np.where(region_num == 1),:,:],axis=(0,1))
hb_data = np.mean(data[np.where(region_num == 2),:,:],axis=(0,1))
X_var = ['PM25_Sim','PM25_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WDIR_Bias','WSPD_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim','PM25_Obs','PM25_Bias']

def get_x_dataset(input_dataset):
    X = np.zeros((len(input_dataset),len(var_sele)+2))
    i = 0
    for var in var_sele:
        X[:,i] = input_dataset[:,var_dict.get(var)]
        i += 1
    X[:,i] = input_dataset[:,var_dict.get('PM2.5_Obs')]
    X[:,i+1] = input_dataset[:,var_dict.get('PM2.5_Bias')]
    return X

writer = pd.ExcelWriter("D:/project/data/BTH/new/origi_analysis.xlsx")

datasetX_bj = pd.DataFrame(get_x_dataset(bj_data),index=np.arange(1,364),columns=X_var) #(363, 21)
datasetX_tj = pd.DataFrame(get_x_dataset(tj_data),index=np.arange(1,364),columns=X_var)
datasetX_hb = pd.DataFrame(get_x_dataset(hb_data),index=np.arange(1,364),columns=X_var)
datasetX_bj = datasetX_bj.round(decimals=2)
datasetX_tj = datasetX_tj.round(decimals=2)
datasetX_hb = datasetX_hb.round(decimals=2)

datasetX_bj.to_excel(writer,'bj_basic')
datasetX_tj.to_excel(writer,'tj_basic')
datasetX_hb.to_excel(writer,'hb_basic')
writer.close()
'''

X = ['PM25_Sim','PM25_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WDIR_Bias','WSPD_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim','PM25_Bias']
sheets = ['bj_basic','tj_basic','hb_basic']
units = ['(ug/$\mathregular{m^3}$)','(ug/$\mathregular{m^3}$)','(ug/$\mathregular{m^3}$)','(ug/$\mathregular{m^3}$)','(ug/$\mathregular{m^3}$)','(ug/$\mathregular{m^3}$)','(ug/$\mathregular{m^3}$)','(ug/$\mathregular{m^3}$)','(%)','(℃)','(°)','(m/s)','(cm)','(%)','(℃)','(m/s)','(cm)','(m)','(W/$\mathregular{m^2}$)','(ug/$\mathregular{m^3}$)']
city_name = ['bj','tj','hb']
city_ch = ['北京','天津','河北']
'''
#京津冀分开作图
j = 0
for city in sheets:
    i = 0
    data = pd.read_excel("D:/project/data/BTH/new/origi_analysis.xlsx", sheet_name=city)
    for x in X:
        fig, axes = plt.subplots(1, 2)
        #1
        labels = ['PM2.5<75','75~115','115~150','150~250','>250']#图例
        box_1, box_2, box_3, box_4, box_5 = data[data.PM25_Obs<75][x], data[(data.PM25_Obs>=75) & (data.PM25_Obs<115)][x], data[(data.PM25_Obs>=115) & (data.PM25_Obs<150)][x], data[(data.PM25_Obs>=150) & (data.PM25_Obs<250)][x], data[data.PM25_Obs>=250][x]
        axes[0].set_title(city_ch[j]+u'不同污染程度下'+x+'值的对比',fontsize=10)#标题，并设定字号大小
        axes[0].boxplot([box_1, box_2, box_3, box_4, box_5],showmeans=True)
        axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_ylabel(x+' '+units[i])
        #2
        labels_season = ['spring','summer','autum','winter']#图例
        box_6, box_7, box_8, box_9= data[x][60:150], data[x][150:240], data[x][240:330], pd.concat([data[x][:60],data[x][330:]],axis=0)
        axes[1].set_title(city_ch[j]+u'不同季节下'+x+'值的对比',fontsize=10)#标题，并设定字号大小
        axes[1].boxplot([box_6, box_7, box_8, box_9],showmeans=True)
        axes[1].set_xticklabels(labels_season, fontsize=8)
        plt.savefig("D:/project/data/BTH/new/contri_origi_boxplot/"+city_name[j]+'_'+x+".png")
        print(city_name[j]+'_'+x+".png saved!")
        i += 1
    j += 1
'''
#京津冀同一个变量连在一起
k = 0 #for units
for x in X:
    fig, axes = plt.subplots(1, 6,figsize=(20,8))
    j = 0 #for city name in the title 
    i = 0 #for axes
    for city in sheets:
        data = pd.read_excel("D:/project/data/BTH/new/origi_analysis.xlsx", sheet_name=city)
        #1
        labels = ['PM2.5<75','75~115','115~150','150~250','>250']#图例
        box_1, box_2, box_3, box_4, box_5 = data[data.PM25_Obs<75][x], data[(data.PM25_Obs>=75) & (data.PM25_Obs<115)][x], data[(data.PM25_Obs>=115) & (data.PM25_Obs<150)][x], data[(data.PM25_Obs>=150) & (data.PM25_Obs<250)][x], data[data.PM25_Obs>=250][x]
        axes[i].set_title(city_ch[j]+u'不同污染程度下'+x+'值的对比',fontsize=10)#标题，并设定字号大小
        axes[i].boxplot([box_1, box_2, box_3, box_4, box_5],showmeans=True)
        axes[i].set_xticklabels(labels, fontsize=8)
        axes[i].set_ylabel(x+' '+units[k])
        i += 1
        #2
        labels_season = ['spring','summer','autum','winter']#图例
        box_6, box_7, box_8, box_9= data[x][60:150], data[x][150:240], data[x][240:330], pd.concat([data[x][:60],data[x][330:]],axis=0)
        axes[i].set_title(city_ch[j]+u'不同季节下'+x+'值的对比',fontsize=10)#标题，并设定字号大小
        axes[i].boxplot([box_6, box_7, box_8, box_9],showmeans=True)
        axes[i].set_xticklabels(labels_season, fontsize=8)
        i += 1
        j += 1
    plt.subplots_adjust(wspace=0.35)
    plt.savefig("D:/project/data/BTH/new/contri_origi_boxplot/"+x+".png",bbox_inches='tight')
    print(x+".png saved!")
    k += 1