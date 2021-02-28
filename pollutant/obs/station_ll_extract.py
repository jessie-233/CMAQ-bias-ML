import numpy as np


with open('D:/project/data/pollutant/obs/monitorij_cn27v2.txt','r',encoding='utf-8') as f:
    sll = np.zeros((1644,2), dtype=np.float)
    i = 0
    for line in f:
        fac2 = line.strip().split('	')[3:5] #经度，纬度
        fac2 = np.array(fac2,dtype=float)
        sll[i,0] = fac2[0]   # lon
        sll[i,1] = fac2[1]   # lat
        i += 1 #共有1644个站点 最终i=1644
np.savetxt('D:/project/data/pollutant/obs/station_ll.csv',sll,delimiter=',')