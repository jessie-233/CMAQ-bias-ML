# PRE 降水
# RHU 相对湿度
# WIN  风速风向
# TEM  温度

import numpy as np
import os
import time

time_start=time.time()
days_list_pernum = [31,28,31,30,31,30,31,31,30,31,30,31]          
after_tran_path = 'E:/project/data/2015/mete/obs/cma/after_trans.csv' 
after_tran_data = np.loadtxt(after_tran_path,delimiter=',',skiprows=1)
#print(after_tran_data)

mete_data_avgday_yearly = np.zeros((232,182,days_list_pernum[0],5), dtype=np.float)#232列，182行，365天，5个气象因子
for month in range(12):
    month_str = str(month+1).zfill(2)
    days = days_list_pernum[month]
    mete_data_avgday = np.zeros((232,182,days,5), dtype=np.float)#232列，182行，365天，5个气象因子
    nsite = np.zeros((232,182,days,5), dtype=np.float)#某站点是否有数据
    
    # 降水
    fp_pre = 'E:/project/data/2015/mete/obs/cma/data/SURF_CLI_CHN_MUL_DAY-PRE-13011-2015'+month_str+'.TXT' 
    pre_data = np.loadtxt(fp_pre)
    for i in range(len(pre_data)): #每条数据
        day = int(pre_data[i,6]) - 1
        site_number = pre_data[i,0]
        pos = np.where(after_tran_data==site_number)
        #print(pos[0][0])
        col = int(after_tran_data[pos[0][0],3])
        row = int(after_tran_data[pos[0][0],4])
        #print(col,row)
        nsite[col,row,day,0] += 1
        if pre_data[i,9] == 32700:
            mete_data_avgday[col,row,day,0] = 0 
        else:
            mete_data_avgday[col,row,day,0] = pre_data[i,9] / 100

    # 相对湿度
    fp_rhu = 'E:/project/data/2015/mete/obs/cma/data/SURF_CLI_CHN_MUL_DAY-RHU-13003-2015'+month_str+'.TXT' 
    rhu_data = np.loadtxt(fp_rhu)
    for i in range(len(rhu_data)):
        day = int(rhu_data[i,6]) - 1
        site_number = rhu_data[i,0]
        pos = np.where(after_tran_data==site_number)
        col = int(after_tran_data[pos[0][0],3])
        row = int(after_tran_data[pos[0][0],4])
        nsite[col,row,day,1] += 1
        mete_data_avgday[col,row,day,1] = rhu_data[i,7] 
    
    # 风速与风向
    fp_win = 'E:/project/data/2015/mete/obs/cma/data/SURF_CLI_CHN_MUL_DAY-WIN-11002-2015'+month_str+'.TXT' 
    win_data = np.loadtxt(fp_win)
    for i in range(len(win_data)):
        day = int(win_data[i,6]) - 1
        site_number = win_data[i,0]
        pos = np.where(after_tran_data==site_number)
        col = int(after_tran_data[pos[0][0],3])
        row = int(after_tran_data[pos[0][0],4])
        nsite[col,row,day,2] += 1
        nsite[col,row,day,3] += 1
        mete_data_avgday[col,row,day,2] = win_data[i,7] / 10        
        mete_data_avgday[col,row,day,3] = (win_data[i,11] - 1) * 22.5
    
    # 温度
    fp_tem = 'E:/project/data/2015/mete/obs/cma/data/SURF_CLI_CHN_MUL_DAY-TEM-12001-2015'+month_str+'.TXT' 
    tem_data = np.loadtxt(fp_tem)
    for i in range(len(tem_data)):
        day = int(tem_data[i,6]) - 1
        site_number = tem_data[i,0]
        pos = np.where(after_tran_data==site_number)
        col = int(after_tran_data[pos[0][0],3])
        row = int(after_tran_data[pos[0][0],4])
        nsite[col,row,day,4] += 1
        mete_data_avgday[col,row,day,4] = tem_data[i,7] / 10        
    #print(mete_data_avgday[136,115,:,:])
    # 没有站点的数据变为-999.
    data_val = 0
    for col in range(232):
        for row in range(182):
            for day in range(days):
                for i in range(5):
                    #print(nsite[col,row,day,i])
                    if nsite[col,row,day,i] == 0:
                        mete_data_avgday[col,row,day,i] = -999. 
                    else:
                        data_val += 1
    print(data_val) 
    if month == 0:
        mete_data_avgday_yearly = mete_data_avgday
    else:
        mete_data_avgday_yearly = np.append(mete_data_avgday_yearly,mete_data_avgday,axis = 2)
    print(month)

np.save('E:/project/data/2015/mete/obs/cma/m_obs_daily.npy',mete_data_avgday_yearly)

time_end=time.time()
print('totally time cost',time_end-time_start)