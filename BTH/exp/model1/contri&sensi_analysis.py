import numpy as np
import pandas as pd
import copy

np.set_printoptions(suppress=True) #抑制使用对小数的科学记数法
np.set_printoptions(precision=2) #设置输出的精度

all_vars = ['PM2.5_Bias', 'PM10_Bias', 'NO2_Bias', 'SO2_Bias', 'O3_Bias', 'CO_Bias', 'PM2.5_Obs', 'PM10_Obs', 'NO2_Obs', 'SO2_Obs', 'O3_Obs', 'CO_Obs', 'PM2.5_Sim','PM10_Sim','NO2_Sim','SO2_Sim','O3_Sim','CO_Sim', 'RH_Bias', 'TEM_Bias', 'WSPD_Bias', 'WDIR_Bias', 'PRE_Bias', 'RH_Obs', 'TEM_Obs', 'WSPD_Obs', 'WDIR_Obs', 'PRE_Obs', 'PBLH_Sim', 'SOLRAD_Sim','RH_Sim','TEM_Sim','WSPD_Sim','WDIR_Sim','PRE_Sim','PM2.5_Bias_ystd','NO2_Bias_ystd','RH_Bias_ystd','O3_Bias_ystd','SO2_Bias_ystd','WSPD_Bias_ystd','NO2_Obs_ystd','O3_Obs_ystd']
var_dict = {'PM2.5_Bias':0, 'PM10_Bias':1, 'NO2_Bias':2, 'SO2_Bias':3, 'O3_Bias':4, 'CO_Bias':5, 'PM2.5_Obs':6, 'PM10_Obs':7, 'NO2_Obs':8, 'SO2_Obs':9, 'O3_Obs':10, 'CO_Obs':11, 'PM2.5_Sim':12, 'PM10_Sim':13, 'NO2_Sim':14, 'SO2_Sim':15, 'O3_Sim':16, 'CO_Sim':17, 'RH_Bias':18, 'TEM_Bias':19, 'WSPD_Bias':20, 'WDIR_Bias':21, 'PRE_Bias':22, 'RH_Obs':23, 'TEM_Obs':24, 'WSPD_Obs':25, 'WDIR_Obs':26, 'PRE_Obs':27, 'PBLH_Sim':28, 'SOLRAD_Sim':29, 'RH_Sim':30, 'TEM_Sim':31, 'WSPD_Sim':32, 'WDIR_Sim':33, 'PRE_Sim':34, 'PM2.5_Bias_ystd':35, 'NO2_Bias_ystd':36, 'RH_Bias_ystd':37, 'O3_Bias_ystd':38, 'SO2_Bias_ystd':39, 'WSPD_Bias_ystd':40, 'NO2_Obs_ystd':41, 'O3_Obs_ystd':42}
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','NO2_Obs','SO2_Obs','O3_Obs','RH_Bias','TEM_Bias','WDIR_Bias','WSPD_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']
BTH_data = np.mean(np.load("D:/project/data/BTH/dataset_BTH.npy"),axis=0) #363, 43)
contribution_data = np.load("D:/project/data/BTH/exp/model1/contribution_result_BTH.npy") #(100, 363, 19) 12、49都是nan，其余都有值
sensitivity_data = np.load("D:/project/data/BTH/exp/model1/sensitivity_result_BTH.npy") #(100, 363, 19) 12、49都是nan，其余个别有nan
# for num in range(2):
#     for row in range(363):
#             print(data[num,row,:])

# for num in range(100):
#     for row in range(363):
#         if np.all(np.isnan(sensitivity_data[num,row,:])):    
#             print(num,row) #12、49都是nan，其余都有值

BTH_contribution_detail = np.nanmean(contribution_data,axis=0) #跳过nan取平均 (363, 19)
BTH_sensitivity_detail = np.nanmean(sensitivity_data,axis=0) #跳过nan取平均 (363, 19)

#制作sheets
#BTH_detail
BTH_contribution_detail_pd = pd.DataFrame(np.concatenate((BTH_contribution_detail,BTH_data[:,var_dict.get('PM2.5_Obs')].reshape((363,1))),axis=1),index=np.arange(1,364),columns=var_sele+['PM2.5_Obs'])
BTH_sensitivity_detail_pd = pd.DataFrame(np.concatenate((BTH_sensitivity_detail,BTH_data[:,var_dict.get('PM2.5_Obs')].reshape((363,1))),axis=1),index=np.arange(1,364),columns=var_sele+['PM2.5_Obs'])

#BTH_combine
BTH_contribution_combine_pd = pd.DataFrame({'PM2.5_Bias_ystd':BTH_contribution_detail_pd.loc[:,'PM2.5_Bias_ystd'],
                               'Pollutants':BTH_contribution_detail_pd.loc[:,'PM2.5_Sim']+BTH_contribution_detail_pd.loc[:,'NO2_Obs']+BTH_contribution_detail_pd.loc[:,'NO2_Bias']+BTH_contribution_detail_pd.loc[:,'SO2_Bias']+BTH_contribution_detail_pd.loc[:,'SO2_Obs'],
                               'Wind':BTH_contribution_detail_pd.loc[:,'WSPD_Bias']+BTH_contribution_detail_pd.loc[:,'WSPD_Obs']+BTH_contribution_detail_pd.loc[:,'WDIR_Bias'],
                               'Precipitation':BTH_contribution_detail_pd.loc[:,'PRE_Bias']+BTH_contribution_detail_pd.loc[:,'PRE_Obs'],
                               'Weather Condition':BTH_contribution_detail_pd.loc[:,'O3_Obs']+BTH_contribution_detail_pd.loc[:,'O3_Bias']+BTH_contribution_detail_pd.loc[:,'TEM_Obs']+BTH_contribution_detail_pd.loc[:,'TEM_Bias']+BTH_contribution_detail_pd.loc[:,'SOLRAD_Sim']+BTH_contribution_detail_pd.loc[:,'PBLH_Sim']+BTH_contribution_detail_pd.loc[:,'RH_Obs']+BTH_contribution_detail_pd.loc[:,'RH_Bias'],
                               'PM2.5_Obs':BTH_data[:,var_dict.get('PM2.5_Obs')]},index = np.arange(1,364))
        
BTH_sensitivity_combine_pd = pd.DataFrame({'PM2.5_Bias_ystd':BTH_sensitivity_detail_pd.loc[:,'PM2.5_Bias_ystd'],
                               'Pollutants':BTH_sensitivity_detail_pd.loc[:,'PM2.5_Sim']+BTH_sensitivity_detail_pd.loc[:,'NO2_Obs']+BTH_sensitivity_detail_pd.loc[:,'NO2_Bias']+BTH_sensitivity_detail_pd.loc[:,'SO2_Bias']+BTH_sensitivity_detail_pd.loc[:,'SO2_Obs'],
                               'Wind':BTH_sensitivity_detail_pd.loc[:,'WSPD_Bias']+BTH_sensitivity_detail_pd.loc[:,'WSPD_Obs']+BTH_sensitivity_detail_pd.loc[:,'WDIR_Bias'],
                               'Precipitation':BTH_sensitivity_detail_pd.loc[:,'PRE_Bias']+BTH_sensitivity_detail_pd.loc[:,'PRE_Obs'],
                               'Weather Condition':BTH_sensitivity_detail_pd.loc[:,'O3_Obs']+BTH_sensitivity_detail_pd.loc[:,'O3_Bias']+BTH_sensitivity_detail_pd.loc[:,'TEM_Obs']+BTH_sensitivity_detail_pd.loc[:,'TEM_Bias']+BTH_sensitivity_detail_pd.loc[:,'SOLRAD_Sim']+BTH_sensitivity_detail_pd.loc[:,'PBLH_Sim']+BTH_sensitivity_detail_pd.loc[:,'RH_Obs']+BTH_sensitivity_detail_pd.loc[:,'RH_Bias'],
                               'PM2.5_Obs':BTH_data[:,var_dict.get('PM2.5_Obs')]},index = np.arange(1,364))

#写入excel
writer = pd.ExcelWriter("D:/project/data/BTH/exp/model1/model1_BTH.xlsx")
BTH_contribution_detail_pd.to_excel(writer,'BTH_contribution_detail',float_format='%.1f')
BTH_sensitivity_detail_pd.to_excel(writer,'BTH_sensitivity_detail',float_format='%.1f')
BTH_contribution_combine_pd.to_excel(writer,'BTH_contribution_combine',float_format='%.1f')
BTH_sensitivity_combine_pd.to_excel(writer,'BTH_sensitivity_combine',float_format='%.1f')
writer.close()
