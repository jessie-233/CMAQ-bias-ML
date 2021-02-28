import os
import numpy as np
import numpy.ma as ma
import time

time_start=time.time()

ns = 1644 #站点数目
days = 31
with open('E:/project/data/2015/pollutant/obs/monitorij_cn27v2.txt','r',encoding='utf-8') as f:
    #文件中的行列号从0开始
    sna = np.zeros((ns,), dtype=np.int)#站点，sna[0]=1001，...
    sid = np.zeros((ns,2), dtype=np.int)#sid[0,0]是第1个站点的列号，sid[0,1]是第1个站点的行号
    sll = np.zeros((ns,2), dtype=np.float)
    i = 0
    for line in f:
        fac = line.strip().split('A')[:1] #站点编号'1001'
        fac = np.array(fac,dtype=int) #将站点编号转换成int 1001
        sna[i] = fac
        fac2 = line.strip().split('	')[5:] #列号，行号 如['136','115']（从0开始）
        fac2 = np.array(fac2,dtype=int)
        sid[i,0] = fac2[0]   # col
        sid[i,1] = fac2[1]   # row
        i += 1 #共有1644个站点 最终i=1644

days_list_pernum = [31,28,31,30,31,30,31,31,30,31,30,31]          
data_avgday_yearly = np.zeros((232,182,days_list_pernum[0],6), dtype=np.float)
      
#对每一个监测点 完善mdata
for month in range(12):
    days = days_list_pernum[month]
    data = np.zeros((232,182,days,24,6), dtype=np.float)  #把中国共分为232个col，182个row（网格） mdata维度: row col 哪天 哪小时 哪种污染物
    nsite = np.zeros((232,182,days,24,6), dtype=np.int)   #网格号某一指标有效值个数
    #print(ns)
    for j in range(ns):
        # 如果没有找到这个站点的txt文件，则跳过
        fn = 'E:/project/data/2015/pollutant/obs/obs2015_'+str(month+1)+'/'+str(sna[j])+'A.txt' #各列污染物顺序是NO2、SO2、O3、PM25、PM10、CO
        if os.path.exists(fn):
            pass 
        else:
            print(fn)
            continue
        a = np.loadtxt(fn)
        # 无效值换为0
        a[a == -999.] = 0 
        data[sid[j,0],sid[j,1],:,:,:] += a[:,1:].reshape((days,24,6))

        #print(data[sid[j,0],sid[j,1],:,:,:])
        for i in range(6):
            for per_day in range(days):
                for per_hour in range(24):
                    if data[sid[j,0],sid[j,1],per_day,per_hour,i] != 0:
                        #print(data[sid[j,0],sid[j,1],per_day,per_hour,i])
                        nsite[sid[j,0],sid[j,1],per_day,per_hour,i] += 1  #某天某小时某网格内有多少站点有有效值；

    data_avgday = np.zeros((232,182,days,6), dtype=np.float)
    for nc in range(232):
        for nr in range(182):
            for per_day in range(days):
                for i in range(6):
                    hour_val = 0
                    for per_hour in range(24):
                        #若此网格内有多个站点，应该取平均，若无站点，置为-999.
                        if nsite[nc,nr,per_day,per_hour,i] > 0:
                            #print(nsite[nc,nr,per_day,per_hour,i])
                            data[nc,nr,per_day,per_hour,i] = data[nc,nr,per_day,per_hour,i]/nsite[nc,nr,per_day,per_hour,i]
                            data_avgday[nc,nr,per_day,i] += data[nc,nr,per_day,per_hour,i]
                            hour_val += 1
                        else:
                            data[nc,nr,per_day,per_hour,i] = -999.
                    if hour_val > 0:
                        data_avgday[nc,nr,per_day,i] = data_avgday[nc,nr,per_day,i] / hour_val
                    else:
                        data_avgday[nc,nr,per_day,i] = -999.
    print(month+1)
    print(data_avgday[136,115,:,1])
    if month == 0:
        data_avgday_yearly = data_avgday
    else:
        data_avgday_yearly = np.append(data_avgday_yearly,data_avgday,axis=2)
    #print(data_avgday_yearly[136,115,:,1])

#print(data_avgday[120:130,100:120,:,:])
time_end=time.time()
print('totally cost',time_end-time_start)
#data_avgday_yearly = ma.masked_array(data_avgday_yearly, data==-999.)  #掩盖无效值 --
np.save('E:/project/data/2015/pollutant/obs/daily.npy',data_avgday_yearly)
#np.save('E:/project/data/2015/pollutant/obs/dayly.txt',data_avgday_yearly)
