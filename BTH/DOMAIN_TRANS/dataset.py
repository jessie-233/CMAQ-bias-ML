import numpy as np
import numpy.ma as ma
import copy
np.set_printoptions(suppress=True) #抑制使用对小数的科学记数法
np.set_printoptions(precision=2) #设置输出的精度
#PM10 Obs数据有问题

#最终data
data = np.zeros((44,363,42)) #第一天的数据都缺少；并且由于有bias_ystd变量，再删去第二天数据

#污染物观测 NO2,SO2,O3,PM25,PM10,CO(mg/m3)
p_obs = np.load('D:/project/data/pollutant/obs/p_obs_daily.npy') #(232, 182, 365, 6)
p_obs = p_obs[:,:,:,:4]
#污染物模拟 PM25,PM10,SO2,NO2,O3,CO(mg/m3)
p_sim = np.load('D:/project/data/pollutant/cmaq/p_sim_daily.npy') #(232, 182, 365, 6)

sim_var = [3, 2, 4, 0, 1, 5]
p_loc = [(154,68),(151,69),(151,70),(156,71),(141,72),(150,72),(137,73),(142,73),(151,73),(137,74),(145,74),(148,74),(157,75),(144,76),(151,76),(153,76),(156,76),(157,76),(150,77),(151,77),(152,77),(158,77),(159,77),(152,78),(151,79),(153,79),(152,81),(153,81),(155,81),(138,82),(140,82),(141,82),(154,82),(155,82),(140,83),(141,83),(150,83),(145,84),(146,84),(147,84),(150,84),(151,84),(152,84),(153,84),(136,86),(139,86),(140,86),(141,87),(139,90),(146,90),(147,90),(150,90),(138,91),(139,91),(143,91),(144,91),(140,92),(140,93),(141,94),(146,94),(147,95)] #78个
m_loc = [(154,67),(150,68),(151,70),(156,71),(141,72),(150,72),(137,73),(141,72),(151,73),(137,73),(145,74),(148,74),(156,76),(144,76),(151,77),(154,76),(156,76),(158,77),(150,77),(151,77),(151,77),(158,77),(158,77),(151,77),(151,79),(154,79),(152,80),(152,80),(155,82),(138,82),(141,82),(141,82),(154,82),(155,82),(141,82),(141,82),(151,83),(146,83),(146,83),(146,83),(151,83),(151,83),(151,83),(153,85),(135,86),(139,85),(139,85),(141,87),(139,90),(146,90),(146,90),(151,89),(139,90),(139,90),(142,91),(145,92),(140,92),(140,92),(142,93),(147,93),(146,95)]#p_loc对应的最邻近气象站点

for loc in p_loc:
    print(loc, np.sum(p_obs[loc[0],loc[1]] == -999.), np.where(p_obs[loc[0],loc[1]] == -999.))

def Fix(col,row,day,item):
    global p_obs
    day_copy1 = copy.deepcopy(day)
    day_copy2 = copy.deepcopy(day)
    min_day = day
    max_day = day
    while p_obs[col,row,day_copy1,item] == -999.0:
        day_copy1 -= 1
        min_day = day_copy1
    while p_obs[col,row,day_copy2,item] == -999.0:
        day_copy2 += 1
        max_day = day_copy2
    p_obs[col,row,day,item] = (p_obs[col,row,min_day,item] + p_obs[col,row,max_day,item]) / 2
    print(p_obs[col,row,day,item])
# (154, 68) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (151, 69) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (151, 70) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (156, 71) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (141, 72) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (150, 72) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (137, 73) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (142, 73) 17 (array([  0,   0,   0,   0, 100, 135, 135, 135, 135, 148, 148, 148, 148,
#        213, 213, 213, 213], dtype=int64), array([0, 1, 2, 3, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
Fix(142,73,100,2)
for item in range(4):
    Fix(142,73,135,item)
    Fix(142,73,148,item)
    Fix(142,73,213,item)

# (151, 73) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (137, 74) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (145, 74) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (148, 74) 12 (array([ 0,  0,  0,  0, 94, 94, 94, 94, 95, 95, 95, 95], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
for item in range(4):
    Fix(148,74,94,item)
    Fix(148,74,95,item)
# (157, 75) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (144, 76) 5 (array([ 0,  0,  0,  0, 12], dtype=int64), array([0, 1, 2, 3, 3], dtype=int64))
Fix(144,76,12,3)
# (151, 76) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (153, 76) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (156, 76) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (157, 76) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (150, 77) 24 (array([  0,   0,   0,   0, 210, 210, 210, 210, 211, 211, 211, 211, 212,212, 212, 212, 213, 213, 213, 213, 214, 214, 214, 214], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,2, 3], dtype=int64))
for item in range(4):
    Fix(150,77,210,item)
    Fix(150,77,211,item)
    Fix(150,77,212,item)
    Fix(150,77,213,item)
    Fix(150,77,214,item)
    
# (151, 77) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (152, 77) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (158, 77) 19 (array([  0,   0,   0,   0, 150, 151, 151, 175, 178, 178, 178, 178, 191,191, 191, 191, 282, 282, 282], dtype=int64), array([0, 1, 2, 3, 1, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],dtype=int64))
Fix(158,77,150,1)
Fix(158,77,151,1)
Fix(158,77,151,2)
Fix(158,77,175,3)
for item in range(4):
    Fix(158,77,178,item)
    Fix(158,77,191,item)
for item in range(3):
    Fix(158,77,282,item)
# (159, 77) 8 (array([  0,   0,   0,   0, 191, 191, 191, 191], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
for item in range(4):
    Fix(159,77,191,item)
# 删(141, 78) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# (152, 78) 9 (array([  0,   0,   0,   0, 299, 299, 299, 299, 314], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=int64))
for item in range(4):
    Fix(152,78,299,item)
Fix(152,78,314,0)
# 删(142, 79) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# 删(145, 79) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# (151, 79) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (153, 79) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# 删(144, 81) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# 删(147, 81) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# 删(149, 81) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# 删(151, 81) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# (152, 81) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (153, 81) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# 删(154, 81) 32 (array([  0,   0,   0,   0, 232, 232, 232, 232, 233, 233, 233, 233, 234,
#        234, 234, 234, 245, 245, 245, 245, 246, 246, 246, 246, 247, 247,
#        247, 247, 248, 248, 248, 248], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
#        2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
# (155, 81) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (138, 82) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (140, 82) 8 (array([  0,   0,   0,   0, 241, 241, 241, 241], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
for item in range(4):
    Fix(140,82,241,item)

# (141, 82) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# 删(144, 82) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# 删(149, 82) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# 删(150, 82) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# 删(152, 82) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# 删(153, 82) 56 (array([ 0,  0,  0,  0,  3,  3,  3,  3,  4,  4,  4,  4, 53, 53, 53, 53, 54,
#        54, 54, 54, 56, 56, 56, 56, 57, 57, 57, 57, 59, 59, 59, 59, 60, 60,
#        60, 60, 61, 61, 61, 61, 63, 63, 63, 63, 64, 64, 64, 64, 65, 65, 65,
#        65, 66, 66, 66, 66], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
#        2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
#        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
# (154, 82) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (155, 82) 20 (array([  0,   0,   0,   0, 112, 112, 112, 112, 113, 113, 113, 113, 114,114, 114, 114, 115, 115, 115, 115], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],dtype=int64))
for item in range(4):
    Fix(155,82,112,item)
    Fix(155,82,113,item)
    Fix(155,82,114,item)
    Fix(155,82,115,item)

# (140, 83) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (141, 83) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (150, 83) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# 删(152, 83) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# 删(144, 84) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# (145, 84) 8 (array([  0,   0,   0,   0, 307, 307, 307, 307], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
for item in range(4):
    Fix(145,84,307,item)
# (146, 84) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (147, 84) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (150, 84) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (151, 84) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (152, 84) 8 (array([ 0,  0,  0,  0, 88, 93, 94, 95], dtype=int64), array([0, 1, 2, 3, 3, 3, 3, 3]dtype=int64))
Fix(152,84,88,3)
Fix(152,84,93,3)
Fix(152,84,94,3)
Fix(152,84,95,3)
# (153, 84) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# 删(143, 85) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# (136, 86) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (139, 86) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (140, 86) 20 (array([  0,   0,   0,   0, 171, 171, 171, 171, 172, 172, 172, 172, 173,173, 173, 173, 244, 244, 244, 244], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],dtype=int64))
for item in range(4):
    Fix(140,86,171,item)
    Fix(140,86,172,item)
    Fix(140,86,173,item)
    Fix(140,86,244,item)

# 删(149, 86) 1460 (array([  0,   0,   0, ..., 364, 364, 364], dtype=int64), array([0, 1, 2, ..., 1, 2, 3], dtype=int64))
# (141, 87) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (139, 90) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (146, 90) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (147, 90) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (150, 90) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (138, 91) 5 (array([ 0,  0,  0,  0, 95], dtype=int64), array([0, 1, 2, 3, 2], dtype=int64))
Fix(138,91,95,2)

# (139, 91) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (143, 91) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (144, 91) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (140, 92) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (140, 93) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (141, 94) 9 (array([  0,   0,   0,   0,  86,  87,  88,  89, 126], dtype=int64), array([0, 1, 2, 3, 2, 2, 2, 2, 3], dtype=int64))
Fix(141,94,86,2)
Fix(141,94,87,2)
Fix(141,94,88,2)
Fix(141,94,89,2)
Fix(141,94,126,3)
# (146, 94) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
# (147, 95) 4 (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))


#气象观测 PRE,RH,WSPD,WDIR,TEM
mete_obs = np.load('D:/project/data/mete/obs/cma/m_obs_daily.npy')#(232, 182, 365, 5)
'''
def Fix_mete(col,row,day,item):
    global mete_obs
    day_copy1 = copy.deepcopy(day)
    day_copy2 = copy.deepcopy(day)
    min_day = day
    max_day = day
    while mete_obs[col,row,day_copy1,item] == -999.0:
        day_copy1 -= 1
        min_day = day_copy1
    while p_obs[col,row,day_copy2,item] == -999.0:
        day_copy2 += 1
        max_day = day_copy2
    p_obs[col,row,day,item] = (p_obs[col,row,min_day,item] + p_obs[col,row,max_day,item]) / 2
    print(p_obs[col,row,day,item])
'''
#check on PRE
for loc in m_loc:
    a = np.sum(mete_obs[loc[0],loc[1],:,0] > 50)
    if a != 0:
        print(loc, a, np.where(mete_obs[loc[0],loc[1],:,0] > 50)) #异常值为327.66
# (154, 67) 1 (array([181], dtype=int64),) 327.66
# (156, 71) 1 (array([181], dtype=int64),) 327.66
mete_obs[154,67,181,0] = (mete_obs[154,67,180,0] + mete_obs[154,67,182,0]) / 2
print(mete_obs[154,67,181,0])
mete_obs[156,71,181,0] = (mete_obs[156,71,180,0] + mete_obs[156,71,182,0]) / 2
print(mete_obs[156,71,181,0])

#check on RH
for loc in m_loc:
    a = np.sum(mete_obs[loc[0],loc[1],:,1] > 100)
    if a != 0:
        print(loc, a, np.where(mete_obs[loc[0],loc[1],:,1] > 100)) #无异常值

#check on WSPD
for loc in m_loc:
    a = np.sum(mete_obs[loc[0],loc[1],:,2] == 3276.6)
    if a != 0:
        print(loc, a, np.where(mete_obs[loc[0],loc[1],:,2] == 3276.6)) #异常值为3276.6
# (137, 73) 5 (array([31, 32, 33, 34, 96], dtype=int64),)
# (151, 89) 1 (array([29], dtype=int64),)
def Fix_WSPD(col,row,day):
    global mete_obs
    day_copy1 = copy.deepcopy(day)
    day_copy2 = copy.deepcopy(day)
    min_day = day
    max_day = day
    while mete_obs[col,row,day_copy1,2] == 3276.6:
        day_copy1 -= 1
        min_day = day_copy1
    while mete_obs[col,row,day_copy2,2] == 3276.6:
        day_copy2 += 1
        max_day = day_copy2
    mete_obs[col,row,day,2] = (mete_obs[col,row,min_day,2] + mete_obs[col,row,max_day,2]) / 2
    print(mete_obs[col,row,day,2])

Fix_WSPD(137, 73,31)
Fix_WSPD(137, 73,32)
Fix_WSPD(137, 73,33)
Fix_WSPD(137, 73,34)
Fix_WSPD(137, 73,96)
Fix_WSPD(151, 89,29)


#check on WDIR
for loc in m_loc:
    a = np.sum(mete_obs[loc[0],loc[1],:,3] > 360)
    if a != 0:
        print(loc, a, np.where(mete_obs[loc[0],loc[1],:,3] > 360)) #异常值为737212.5
# (137, 73) 3 (array([31, 32, 33], dtype=int64),)
def Fix_WDIR(col,row,day):
    global mete_obs
    day_copy1 = copy.deepcopy(day)
    day_copy2 = copy.deepcopy(day)
    min_day = day
    max_day = day
    while mete_obs[col,row,day_copy1,3] == 737212.5:
        day_copy1 -= 1
        min_day = day_copy1
    while mete_obs[col,row,day_copy2,3] == 737212.5:
        day_copy2 += 1
        max_day = day_copy2
    mete_obs[col,row,day,3] = (mete_obs[col,row,min_day,3] + mete_obs[col,row,max_day,3]) / 2
    print(mete_obs[col,row,day,3])

Fix_WDIR(137, 73,31)
Fix_WDIR(137, 73,32)
Fix_WDIR(137, 73,33)

#check on TEM
for loc in m_loc:
    a = np.sum(mete_obs[loc[0],loc[1],:,4] < -20) #or大于40
    if a != 0:
        print(loc, a, np.where(mete_obs[loc[0],loc[1],:,4] < -20)) #无异常值


#气象模拟 RH,TEM,PBLH,SOL_RAD,PRE,WSPD,WDIR
mete_sim = np.load('D:/project/data/mete/cmaq/m_sim_daily.npy')#(232, 182, 365, 7)


for i in range(44):
    data[i,:,0] = (p_sim[p_loc[i][0],p_loc[i][1],2:,0] - p_obs[p_loc[i][0],p_loc[i][1],2:,3]) #PM2.5_Bias
    data[i,:,1] = (p_sim[p_loc[i][0],p_loc[i][1],2:,1] - p_obs[p_loc[i][0],p_loc[i][1],2:,4]) #PM10_Bias
    data[i,:,2] = (p_sim[p_loc[i][0],p_loc[i][1],2:,3] - p_obs[p_loc[i][0],p_loc[i][1],2:,0]) #NO2_Bias
    data[i,:,3] = (p_sim[p_loc[i][0],p_loc[i][1],2:,2] - p_obs[p_loc[i][0],p_loc[i][1],2:,1]) #SO2_Bias
    data[i,:,4] = (p_sim[p_loc[i][0],p_loc[i][1],2:,4] - p_obs[p_loc[i][0],p_loc[i][1],2:,2]) #O3_Bias
    data[i,:,5] = (p_sim[p_loc[i][0],p_loc[i][1],2:,5] - p_obs[p_loc[i][0],p_loc[i][1],2:,5]) #CO_Bias
    data[i,:,6] = p_obs[p_loc[i][0],p_loc[i][1],2:,3] #PM2.5_Obs
    data[i,:,7] = p_obs[p_loc[i][0],p_loc[i][1],2:,4] #PM10_Obs
    data[i,:,8] = p_obs[p_loc[i][0],p_loc[i][1],2:,0] #NO2_Obs
    data[i,:,9] = p_obs[p_loc[i][0],p_loc[i][1],2:,1] #SO2_Obs
    data[i,:,10] = p_obs[p_loc[i][0],p_loc[i][1],2:,2] #O3_Obs
    data[i,:,11] = p_obs[p_loc[i][0],p_loc[i][1],2:,5] #CO_Obs
    data[i,:,12] = p_sim[p_loc[i][0],p_loc[i][1],2:,0] #PM2.5_Sim
    data[i,:,13] = p_sim[p_loc[i][0],p_loc[i][1],2:,1] #PM10_Sim
    data[i,:,14] = p_sim[p_loc[i][0],p_loc[i][1],2:,3] #NO2_Sim
    data[i,:,15] = p_sim[p_loc[i][0],p_loc[i][1],2:,2] #SO2_Sim
    data[i,:,16] = p_sim[p_loc[i][0],p_loc[i][1],2:,4] #O3_Sim
    data[i,:,17] = p_sim[p_loc[i][0],p_loc[i][1],2:,5] #CO_Sim
    data[i,:,18] = (mete_sim[m_loc[i][0],m_loc[i][1],2:,0] - mete_obs[m_loc[i][0],m_loc[i][1],2:,1]) #RH_Bias
    data[i,:,19] = (mete_sim[m_loc[i][0],m_loc[i][1],2:,1] - mete_obs[m_loc[i][0],m_loc[i][1],2:,4]) #TEM_Bias
    data[i,:,20] = (mete_sim[m_loc[i][0],m_loc[i][1],2:,5] - mete_obs[m_loc[i][0],m_loc[i][1],2:,2]) #WSPD_Bias
    for j in range(2,365):  #WDIR_Bias
        WDIR_Bias = np.abs(mete_sim[m_loc[i][0],m_loc[i][1],j,6] - mete_obs[m_loc[i][0],m_loc[i][1],j,3])
        if WDIR_Bias > 180.:  
            data[i,:,21] = 360. - WDIR_Bias  
        else:
            data[i,:,21] = WDIR_Bias

    data[i,:,22] = (mete_sim[m_loc[i][0],m_loc[i][1],2:,4] - mete_obs[m_loc[i][0],m_loc[i][1],2:,0]) #PRE_Bias
    data[i,:,23] = mete_obs[m_loc[i][0],m_loc[i][1],2:,1] #RH_Obs
    data[i,:,24] = mete_obs[m_loc[i][0],m_loc[i][1],2:,4] #TEM_Obs
    data[i,:,25] = mete_obs[m_loc[i][0],m_loc[i][1],2:,2] #WSPD_Obs
    data[i,:,26] = mete_obs[m_loc[i][0],m_loc[i][1],2:,3] #WDIR_Obs
    data[i,:,27] = mete_obs[m_loc[i][0],m_loc[i][1],2:,0] #PRE_Obs
    data[i,:,28] = mete_sim[m_loc[i][0],m_loc[i][1],2:,2] #PBLH_Sim
    data[i,:,29] = mete_sim[m_loc[i][0],m_loc[i][1],2:,3] #SOLRAD_Sim
    data[i,:,30] = mete_sim[m_loc[i][0],m_loc[i][1],2:,0] #RH_Sim
    data[i,:,31] = mete_sim[m_loc[i][0],m_loc[i][1],2:,1] #TEM_Sim
    data[i,:,32] = mete_sim[m_loc[i][0],m_loc[i][1],2:,5] #WSPD_Sim
    data[i,:,33] = mete_sim[m_loc[i][0],m_loc[i][1],2:,6] #WDIR_Sim
    data[i,:,34] = mete_sim[m_loc[i][0],m_loc[i][1],2:,4] #PRE_Sim

    #风向风速矢量分解，新加6个变量
    data[i,:,35] = mete_obs[m_loc[i][0],m_loc[i][1],2:,2] * np.cos(mete_obs[m_loc[i][0],m_loc[i][1],2:,3] * np.pi / 180) #WIN_N_Obs
    data[i,:,36] = mete_sim[m_loc[i][0],m_loc[i][1],2:,5] * np.cos(mete_sim[m_loc[i][0],m_loc[i][1],2:,6] * np.pi / 180) #WIN_N_Sim
    data[i,:,37] = mete_obs[m_loc[i][0],m_loc[i][1],2:,2] * np.sin(mete_obs[m_loc[i][0],m_loc[i][1],2:,3] * np.pi / 180) #WIN_E_Obs
    data[i,:,38] = mete_sim[m_loc[i][0],m_loc[i][1],2:,5] * np.sin(mete_sim[m_loc[i][0],m_loc[i][1],2:,6] * np.pi / 180) #WIN_E_Sim
    data[i,:,39] = data[i,:,36] - data[i,:,35] #WIN_N_Bias
    data[i,:,40] = data[i,:,38] - data[i,:,37] #WIN_E_Bias
    #前一天的PM25bias
    data[i,0,41] = p_sim[p_loc[i][0],p_loc[i][1],1,0] - p_obs[p_loc[i][0],p_loc[i][1],1,3]
    data[i,1:,41] = data[i,0:362,0]

#修补WSPD_Obs data
data[13,22,25] = 1.7
data[16,322,25] = 3.0
data[20,322,25] = 4.0
data[13,323,25] = 4
data[16,323,25] = 4
data[20,323,25] = 5
data[14,324,25] = 4
#修补WIN_N_Obs data
data[13,22,35] = data[13,22,25] * np.cos(mete_obs[m_loc[13][0],m_loc[13][1],24,3] * np.pi / 180)
data[16,322,35] = data[16,322,35] * np.cos(mete_obs[m_loc[16][0],m_loc[16][1],324,3] * np.pi / 180)
data[20,322,35] = data[20,322,35] * np.cos(mete_obs[m_loc[20][0],m_loc[20][1],324,3] * np.pi / 180)
data[13,323,35] = data[13,323,35] * np.cos(mete_obs[m_loc[13][0],m_loc[13][1],325,3] * np.pi / 180)
data[16,323,35] = data[16,323,35] * np.cos(mete_obs[m_loc[16][0],m_loc[16][1],325,3] * np.pi / 180)
data[20,323,35] = data[20,323,35] * np.cos(mete_obs[m_loc[20][0],m_loc[20][1],325,3] * np.pi / 180)
data[14,324,35] = data[14,324,35] * np.cos(mete_obs[m_loc[14][0],m_loc[14][1],326,3] * np.pi / 180)
#修补WIN_E_Obs data
data[13,22,37] = data[13,22,25] * np.sin(mete_obs[m_loc[13][0],m_loc[13][1],24,3] * np.pi / 180)
data[16,322,37] = data[16,322,35] * np.sin(mete_obs[m_loc[16][0],m_loc[16][1],324,3] * np.pi / 180)
data[20,322,37] = data[20,322,35] * np.sin(mete_obs[m_loc[20][0],m_loc[20][1],324,3] * np.pi / 180)
data[13,323,37] = data[13,323,35] * np.sin(mete_obs[m_loc[13][0],m_loc[13][1],325,3] * np.pi / 180)
data[16,323,37] = data[16,323,35] * np.sin(mete_obs[m_loc[16][0],m_loc[16][1],325,3] * np.pi / 180)
data[20,323,37] = data[20,323,35] * np.sin(mete_obs[m_loc[20][0],m_loc[20][1],325,3] * np.pi / 180)
data[14,324,37] = data[14,324,35] * np.sin(mete_obs[m_loc[14][0],m_loc[14][1],326,3] * np.pi / 180)
#修补WIN_N_Bias data
data[13,22,39] = data[13,22,36] - data[13,22,35]
data[16,322,39] = data[16,322,36] - data[16,322,35]
data[20,322,39] = data[20,322,36] - data[20,322,35]
data[13,323,39] = data[13,323,36] - data[13,323,35]
data[16,323,39] = data[16,323,36] - data[16,323,35]
data[20,323,39] = data[20,323,36] - data[20,323,35]
data[14,324,39] = data[14,324,36] - data[14,324,35]
#修补WIN_E_Bias data
data[13,22,40] = data[13,22,38] - data[13,22,37]
data[16,322,40] = data[16,322,38] - data[16,322,37]
data[20,322,40] = data[20,322,38] - data[20,322,37]
data[13,323,40] = data[13,323,38] - data[13,323,37]
data[16,323,40] = data[16,323,38] - data[16,323,37]
data[20,323,40] = data[20,323,38] - data[20,323,37]
data[14,324,40] = data[14,324,38] - data[14,324,37]

print(data.shape) #(44, 363, 42)
data = np.around(data, 2) #都保留2位小数
np.save("D:/project/data/BTH/new/dataset_abs.npy", data)

#修改风向异常值
data = np.load("D:/project/data/BTH/new/dataset_abs.npy")
for i in range(44):
    for j in range(363):  #WDIR_Bias
        WDIR_Bias = np.abs(data[i,j,33] -data[i,j,26])
        if WDIR_Bias > 180.:  
            data[i,j,21] = 360. - WDIR_Bias  
        else:
            data[i,j,21] = WDIR_Bias
data = np.around(data, 2) #都保留2位小数
np.save("D:/project/data/BTH/new/dataset_abs_corr.npy", data)

#修改风速异常值
data[13,324,25] = (data[13,323,25] + data[13,325,25]) / 2
for i in range(44):
    data[i,:,20] = data[i,:,32] - data[i,:,25]
'''