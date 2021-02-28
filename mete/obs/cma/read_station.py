import numpy as np
import re
import os

##形成列表：站号-经度（小数）-纬度（小数）
def unit_transfer(lat:int):
    m = lat % 100 #分
    n = int(lat / 100) #度
    return format((n + m / 60), '.3f')

ns = 698
info = np.zeros((ns,3), dtype=np.float)#站号，纬度，经度
path = 'D:/project/data/mete/cma/data'
dirs = os.listdir(path)
i = 0
print(dirs)
for dir in dirs:
    with open('D:/project/data/mete/cma/data/'+ dir,'r',encoding='utf-8') as f:
        for line in f:
            fac = re.split(r"[ ]+", line)
            if np.array(fac[0],dtype=int) not in info[:,0]:
                info[i,0] = np.array(fac[0],dtype=int) #将站点编号转换成int
                info[i,1] = unit_transfer(np.array(fac[1],dtype=int))   # 纬度
                info[i,2] = unit_transfer(np.array(fac[2],dtype=int))   # 经度
                i += 1 #共有698个站点 最终i=698
        print(i)
np.savetxt("D:/project/data/mete/cma/before_trans.csv", info, delimiter=',')


