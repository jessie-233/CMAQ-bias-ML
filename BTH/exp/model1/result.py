import numpy as np
import pandas as pd

contribution_detail = pd.read_excel('D:/project/data/BTH/exp/model1/model1_BTH.xlsx',sheet_name='BTH_contribution_detail')
del contribution_detail['PM2.5_Obs']
contribution_detail_result = contribution_detail.groupby(['season','PM_Level']).mean()

sensitivity_detail = pd.read_excel('D:/project/data/BTH/exp/model1/model1_BTH.xlsx',sheet_name='BTH_sensitivity_detail')
del sensitivity_detail['PM2.5_Obs']
sensitivity_detail_result = sensitivity_detail.groupby(['season','PM_Level']).mean()

contribution_combine = pd.read_excel('D:/project/data/BTH/exp/model1/model1_BTH.xlsx',sheet_name='BTH_contribution_combine')
del contribution_combine['PM2.5_Obs']
contribution_combine_result = contribution_combine.groupby(['season','PM_Level']).mean()

sensitivity_combine = pd.read_excel('D:/project/data/BTH/exp/model1/model1_BTH.xlsx',sheet_name='BTH_sensitivity_combine')
del sensitivity_combine['PM2.5_Obs']
sensitivity_combine_result = sensitivity_combine.groupby(['season','PM_Level']).mean()

writer = pd.ExcelWriter("D:/project/data/BTH/exp/model1/final_BTH.xlsx")
contribution_detail_result.to_excel(writer,'BTH_contribution_detail',float_format='%.1f')
contribution_combine_result.to_excel(writer,'BTH_contribution_combine',float_format='%.1f')
sensitivity_detail_result.to_excel(writer,'BTH_sensitivity_detail',float_format='%.1f')
sensitivity_combine_result.to_excel(writer,'BTH_sensitivity_combine',float_format='%.1f')
writer.close()


















