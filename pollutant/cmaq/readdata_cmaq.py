import numpy as np
import netCDF4 as nc
import os


pollutant_names = ['PM25_TOT','PM10','SO2','NO2','O3','CO']  
mete_names = ['RH','SFC_TMP','PBLH','SOL_RAD','precip','WSPD10','WDIR10']
days_list_pernum = [31,28,31,30,31,30,31,31,30,31,30,31]          

pollutant_data_avgday_yearly = np.zeros((232,182,days_list_pernum[0],6), dtype=np.float)#232列，182行，12个月，31天，24小时，6个污染物
mete_data_avgday_yearly = np.zeros((232,182,days_list_pernum[0],7), dtype=np.float)#232列，182行，12个月，31天，24小时，7个气象因子
for month in range(12):
    days = days_list_pernum[month]
    pollutant_data_avgday = np.zeros((232,182,days,6), dtype=np.float)#232列，182行，365天，24小时，6个污染物
    mete_data_avgday = np.zeros((232,182,days,7), dtype=np.float)#232列，182行，365天，24小时，7个气象因子
    pollutant_data = np.zeros((232,182,days,24,6), dtype=np.float)#232列，182行，365天，24小时，6个污染物
    mete_data = np.zeros((232,182,days,24,7), dtype=np.float)#232列，182行，365天，24小时，7个气象因子    
    f = nc.Dataset('F:/2015cmaq/2015_'+str(month+1)+'.nc')
    for day in range(days):
        for pollutant in range(6):
            var = pollutant_names[pollutant]
            pollutant_data[:,:,day,:,pollutant] = f.variables[var][day*24:day*24+24,0,:,:].transpose(2,1,0) 
        for mete in range(7):
            var = mete_names[mete]
            mete_data[:,:,day,:,mete] = f.variables[var][day*24:day*24+24,0,:,:].transpose(2,1,0)
    # 所有指标值取平均
    pollutant_data_avgday = np.mean(pollutant_data,axis=3) #求日均值(232,182,31,24,6)变为(232,182,31,6)
    mete_data_avgday = np.mean(mete_data,axis=3)
    # 获取风向日均值和降水日均值  
    for col in range(232):
        for row in range(182):
            for day in range(days):
                index = np.argmax(mete_data[col,row,day,:,5],axis=0)
                mete_data_avgday[col,row,day,6] = mete_data[col,row,day,index,6]
                mete_data_avgday[col,row,day,4] = np.sum(mete_data[col,row,day,:,4])

    # 将数据接在其他月份数据上
    if month == 0:
        pollutant_data_avgday_yearly = pollutant_data_avgday
        mete_data_avgday_yearly = mete_data_avgday
        print(pollutant_data_avgday_yearly.shape)
        print(mete_data_avgday_yearly[136,115,:,4])
        print(mete_data_avgday_yearly[136,115,:,6])
    else:
        pollutant_data_avgday_yearly = np.append(pollutant_data_avgday_yearly,pollutant_data_avgday,axis=2)
        mete_data_avgday_yearly = np.append(mete_data_avgday_yearly,mete_data_avgday,axis=2)
    print(month+1)

#np.save('F:/2015cmaq/p_sim_hourly.npy',pollutant_data)
#np.save('F:/2015cmaq/m_sim_hourly.npy',mete_data)
np.save('F:/2015cmaq/p_sim_daily.npy',pollutant_data_avgday_yearly)
np.save('F:/2015cmaq/m_sim_daily.npy',mete_data_avgday_yearly)
