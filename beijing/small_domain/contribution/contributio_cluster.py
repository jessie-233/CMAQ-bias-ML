import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

contrib_dataset = np.load("D:/project/data/beijing/small_domain/contribution/contribution_analysis.npy") #(365, 21)各个特征的单位都是%贡献度，因此不需标准化
var_sele = ['PM2.5_Sim','PM2.5_Bias_ystd','NO2_Bias','SO2_Bias','O3_Bias','CO_Bias','NO2_Obs','SO2_Obs','O3_Obs','CO_Obs','RH_Bias','TEM_Bias','WIN_N_Bias','WIN_E_Bias','PRE_Bias','RH_Obs','TEM_Obs','WSPD_Obs','PRE_Obs','PBLH_Sim','SOLRAD_Sim']
var_dataset = np.load("D:/project/data/beijing/small_domain/var_sele_dataset.npy") #(365,22) 前21列是X变量，最后一列是Y:‘PM2.5_Bias’



##Affinity Propagation算法
def APCluster(preference=-2000):
    global contrib_dataset,var_sele,var_dataset
    writer = pd.ExcelWriter('D:/project/data/beijing/small_domain/contribution/cluster_result/APCluster/'+'APCluster_'+str(preference)+'.xlsx')
    APmodel = AffinityPropagation(preference=preference)
    APmodel.fit(contrib_dataset)
    #制作stat
    r1 = pd.Series(APmodel.labels_).value_counts()
    r2 = pd.DataFrame(APmodel.cluster_centers_indices_)
    r0 = pd.Series(r1 / sum(r1) * 100).round(decimals=2)
    stat = pd.concat([r2,r1,r0], axis=1)
    stat.columns = ['center_indices','sample_number','sample_percent']
    indx = ['cluster'+str(i) for i in range(len(APmodel.cluster_centers_indices_))]
    stat.index = indx
    stat.to_excel(writer,'basic_stat')    
    #制作contrib的平均值、标准差stat2
    stat2 = pd.DataFrame()
    contrib_dataset_pd = pd.DataFrame(contrib_dataset,index=range(len(contrib_dataset)),columns=var_sele)
    contrib_dataset_revised = pd.concat([contrib_dataset_pd,pd.Series(APmodel.labels_,index=contrib_dataset_pd.index)], axis=1)
    contrib_dataset_revised.columns = list(contrib_dataset_pd.columns) + ['cluster_label']
    for i in range(len(APmodel.cluster_centers_indices_)):
        seleted = contrib_dataset_revised[contrib_dataset_revised.cluster_label == i].iloc[:,:len(contrib_dataset_pd.columns)] #去掉最后一列‘cluster_label’
        mean = seleted.describe().iloc[1] #(21,)平均值 第2行
        std = seleted.describe().iloc[2] #标准差 第3行
        temp = pd.DataFrame(pd.concat([mean,std],axis=0)).T
        stat2 = pd.concat([stat2,temp],axis=0)
    stat2['sum_std'] = stat2.iloc[:,len(var_sele):].apply(lambda x: x.sum(), axis=1) #标准差的和
    stat2 = stat2.round(decimals=2)
    stat2.columns = [var+'contrib_mean' for var in var_sele] + [var+'contrib_std' for var in var_sele] + ['sum_std']
    stat2.index = stat.index
    stat2.to_excel(writer,'cluster_contrib_stat')
    #聚类中心
    stat3 = pd.DataFrame(APmodel.cluster_centers_,index=stat.index,columns=var_sele)
    stat3 = stat3.round(decimals=2)
    stat3.to_excel(writer,'cluster_centre')
    #制作源数据的聚类统计
    stat4 = pd.DataFrame()
    var_dataset_pd = pd.DataFrame(var_dataset,index=range(len(var_dataset)),columns=var_sele+['PM2.5_Bias'])
    var_dataset_revised = pd.concat([var_dataset_pd,pd.Series(APmodel.labels_,index=var_dataset_pd.index)], axis=1)
    var_dataset_revised.columns = list(var_dataset_pd.columns) + ['cluster_label']
    for i in range(len(APmodel.cluster_centers_indices_)):
        seleted = var_dataset_revised[var_dataset_revised.cluster_label == i].iloc[:,:len(var_dataset_pd.columns)] #去掉最后一列‘cluster_label’
        mean = seleted.describe().iloc[1] #(22,)平均值 第2行
        std = seleted.describe().iloc[2]  #(22,) 标准差 第3行
        temp = pd.DataFrame(pd.concat([mean,std],axis=0)).T
        stat4 = pd.concat([stat4,temp],axis=0)
    stat4['sum_std'] = stat4.iloc[:,len(var_sele):].apply(lambda x: x.sum(), axis=1) #标准差的和
    stat4 = stat4.round(decimals=2)
    stat4.columns = [var+'_mean' for var in var_dataset_pd.columns] + [var+'_std' for var in var_dataset_pd.columns] + ['sum_std']
    stat4.index = stat.index
    stat4.to_excel(writer,'cluster_orign_stat')
    #制作date—label
    stat5 = pd.DataFrame(APmodel.labels_)
    stat5.to_excel(writer,'date—label')
    writer.close()
    #可视化1：基于contribution
    tsne = TSNE(random_state=105)
    tsne.fit_transform(contrib_dataset_pd)
    tsne = pd.DataFrame(tsne.embedding_,index=contrib_dataset_pd.index)
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    #不同类别用不同颜色和样式绘图
    for i in range(len(APmodel.cluster_centers_indices_)):
        d = tsne[contrib_dataset_revised['cluster_label'] == i]  #找出聚类类别为i的数据对应的降维结果
        plt.scatter(d[0],d[1],s=50)
    plt.savefig('D:/project/data/beijing/small_domain/contribution/cluster_result/APCluster/'+'APCluster_'+str(preference)+'.png')
    plt.close()
    #plt.show()

    #可视化2：基于源数据var_dataset
    tsne2 = TSNE(random_state=236)
    tsne2.fit_transform(var_dataset_pd)
    tsne2 = pd.DataFrame(tsne2.embedding_,index=var_dataset_pd.index)
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    #不同类别用不同颜色和样式绘图
    for i in range(len(APmodel.cluster_centers_)):
        d = tsne2[var_dataset_revised['cluster_label'] == i]  #找出聚类类别为i的数据对应的降维结果
        plt.scatter(d[0],d[1],s=50)
    plt.savefig('D:/project/data/beijing/small_domain/contribution/cluster_result/APCluster/'+'APCluster_'+str(preference)+'_var'+'.png')
    plt.close()

#APCluster(preference=-1000)
APCluster(preference=-1500)
APCluster(preference=-2000)
APCluster(preference=-2500)
APCluster(preference=-3000)


"""
-1000
[  0  12  45  55  83  95 117 142 151 182 206 246 272 277 309 326 330 334
 342]
81
-1500
[ 45  55  61  95 106 110 142 151 175 246 272 330]
65
-2000
[  0  45  55 106 110 142 206 272]
104
-2500
[  0  45  55 110 142 155 272]
118
-3000
[  0  45  55 110 142 206 272]
137
"""

"""
##Mean Shift算法(失败，只分了一类)
def MSCluster():
    global dataset,var_sele
    writer = pd.ExcelWriter('D:/project/data/beijing/small_domain/contribution/cluster_result/MSCluster/'+'MSCluster'+'.xlsx')
    bandwidth = estimate_bandwidth(dataset, quantile=0.5, n_samples=30)
    MSmodel = MeanShift(bandwidth=bandwidth,bin_seeding = True)
    MSmodel.fit(dataset)

    #制作stat
    r1 = pd.Series(MSmodel.labels_).value_counts()
    r0 = pd.Series(r1 / sum(r1) * 100).round(decimals=2)
    stat = pd.concat([r1,r0], axis=1)
    stat.columns = ['sample_number','sample_percent']
    indx = ['cluster'+str(i) for i in range(len(MSmodel.cluster_centers_))]
    stat.index = indx
    stat.to_excel(writer,'stat')
    print(stat)
    
    #制作dataset_revised
    dataset_pd = pd.DataFrame(dataset,index=range(len(dataset)),columns=var_sele)
    dataset_revised = pd.concat([dataset_pd,pd.Series(MSmodel.labels_,index=dataset_pd.index)], axis=1)
    dataset_revised.columns = list(dataset_pd.columns) + ['cluster_label']
    dataset_revised.to_excel(writer,'clustered')
    #print(dataset_revised)

    #聚类中心
    stat3 = pd.DataFrame(MSmodel.cluster_centers_,index=stat.index,columns=var_sele)
    stat3 = stat3.round(decimals=2)
    stat3.to_excel(writer,'stat3')
    writer.close()

    #可视化
    tsne = TSNE(random_state=105)
    tsne.fit_transform(dataset_pd)
    tsne = pd.DataFrame(tsne.embedding_,index=dataset_pd.index)
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    #不同类别用不同颜色和样式绘图
    for i in range(len(MSmodel.cluster_centers_)):
        d = tsne[dataset_revised['cluster_label'] == i]  #找出聚类类别为i的数据对应的降维结果
        plt.scatter(d[0],d[1],s=50)
    plt.savefig('D:/project/data/beijing/small_domain/contribution/cluster_result/MSCluster/'+'MSCluster'+'.png')
    plt.show()
"""
"""
##K-Means算法
def KMCluster(n_clusters):
    global contrib_dataset,var_sele,var_dataset
    writer = pd.ExcelWriter('D:/project/data/beijing/small_domain/contribution/cluster_result/KMCluster/'+'KMCluster_'+str(n_clusters)+'.xlsx')
    KMmodel = KMeans(n_clusters=n_clusters,max_iter=500)
    KMmodel.fit(contrib_dataset)
    #制作stat
    r1 = pd.Series(KMmodel.labels_).value_counts()
    r0 = pd.Series(r1 / sum(r1) * 100).round(decimals=2)
    stat = pd.concat([r1,r0], axis=1)
    stat.columns = ['sample_number','sample_percent']
    indx = ['cluster'+str(i) for i in range(len(KMmodel.cluster_centers_))]
    stat.index = indx
    stat.to_excel(writer,'basic_stat')
    print(stat)
    #制作contrib的平均值、标准差stat2
    stat2 = pd.DataFrame()
    contrib_dataset_pd = pd.DataFrame(contrib_dataset,index=range(len(contrib_dataset)),columns=var_sele)
    contrib_dataset_revised = pd.concat([contrib_dataset_pd,pd.Series(KMmodel.labels_,index=contrib_dataset_pd.index)], axis=1)
    contrib_dataset_revised.columns = list(contrib_dataset_pd.columns) + ['cluster_label']
    for i in range(len(KMmodel.cluster_centers_)):
        seleted = contrib_dataset_revised[contrib_dataset_revised.cluster_label == i].iloc[:,:len(contrib_dataset_pd.columns)] #去掉最后一列‘cluster_label’
        mean = seleted.describe().iloc[1] #(21,)平均值 第2行
        std = seleted.describe().iloc[2] #标准差 第3行
        temp = pd.DataFrame(pd.concat([mean,std],axis=0)).T
        stat2 = pd.concat([stat2,temp],axis=0)
    stat2['sum_std'] = stat2.iloc[:,len(var_sele):].apply(lambda x: x.sum(), axis=1) #标准差的和
    stat2 = stat2.round(decimals=2)
    stat2.columns = [var+'contrib_mean' for var in var_sele] + [var+'contrib_std' for var in var_sele] + ['sum_std']
    stat2.index = stat.index
    stat2.to_excel(writer,'cluster_contrib_stat')
    #聚类中心（KM算法的聚类中心就是本聚类的平均值）
    stat3 = pd.DataFrame(KMmodel.cluster_centers_,index=stat.index,columns=var_sele)
    stat3 = stat3.round(decimals=2)
    stat3.to_excel(writer,'cluster_centre')
    #制作源数据的聚类统计
    stat4 = pd.DataFrame()
    var_dataset_pd = pd.DataFrame(var_dataset,index=range(len(var_dataset)),columns=var_sele+['PM2.5_Bias'])
    var_dataset_revised = pd.concat([var_dataset_pd,pd.Series(KMmodel.labels_,index=var_dataset_pd.index)], axis=1)
    var_dataset_revised.columns = list(var_dataset_pd.columns) + ['cluster_label']
    for i in range(len(KMmodel.cluster_centers_)):
        seleted = var_dataset_revised[var_dataset_revised.cluster_label == i].iloc[:,:len(var_dataset_pd.columns)] #去掉最后一列‘cluster_label’
        mean = seleted.describe().iloc[1] #(22,)平均值 第2行
        std = seleted.describe().iloc[2]  #(22,) 标准差 第3行
        temp = pd.DataFrame(pd.concat([mean,std],axis=0)).T
        stat4 = pd.concat([stat4,temp],axis=0)
    stat4['sum_std'] = stat4.iloc[:,len(var_sele):].apply(lambda x: x.sum(), axis=1) #标准差的和
    stat4 = stat4.round(decimals=2)
    stat4.columns = [var+'_mean' for var in var_dataset_pd.columns] + [var+'_std' for var in var_dataset_pd.columns] + ['sum_std']
    stat4.index = stat.index
    stat4.to_excel(writer,'cluster_orign_stat')
    #制作date—label
    stat5 = pd.DataFrame(KMmodel.labels_)
    stat5.to_excel(writer,'date—label')
    writer.close()
    #可视化1：基于contribution
    tsne = TSNE(random_state=105)
    tsne.fit_transform(contrib_dataset_pd)
    tsne = pd.DataFrame(tsne.embedding_,index=contrib_dataset_pd.index)
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    #不同类别用不同颜色和样式绘图
    for i in range(len(KMmodel.cluster_centers_)):
        d = tsne[contrib_dataset_revised['cluster_label'] == i]  #找出聚类类别为i的数据对应的降维结果
        plt.scatter(d[0],d[1],s=50)
    plt.savefig('D:/project/data/beijing/small_domain/contribution/cluster_result/KMCluster/'+'KMCluster_'+str(n_clusters)+'.png')
    plt.close()
    #plt.show()

    #可视化2：基于源数据var_dataset
    tsne2 = TSNE(random_state=236)
    tsne2.fit_transform(var_dataset_pd)
    tsne2 = pd.DataFrame(tsne2.embedding_,index=var_dataset_pd.index)
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    #不同类别用不同颜色和样式绘图
    for i in range(len(KMmodel.cluster_centers_)):
        e = tsne2[var_dataset_revised['cluster_label'] == i]  #找出聚类类别为i的数据对应的降维结果
        plt.scatter(e[0],e[1],s=50)
    plt.savefig('D:/project/data/beijing/small_domain/contribution/cluster_result/KMCluster/'+'KMCluster_'+str(n_clusters)+'_var'+'.png')
    plt.close()
    #plt.show()
for i in range(4,8):
    KMCluster(i)
"""
# 问题：
## 1.使每个方法的聚类颜色一样？旁边标注上cluster1？
# 2.每个方法分出来的类是否相近？：看图、看类的均值
# 3.每个类的pattern：主要的贡献变量？做直方图/累计面积图
## 4.每个类对应的输入变量情况pattern：求算每类中各变量均值、方差，做直方图