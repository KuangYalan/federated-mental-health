import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

matplotlib.rcParams['font.family']='SimHei'     #将字体设置为黑体'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

def convert2dic(a,b):
    d = { }
    if len(a) == len(b):
        for i in range(len(a)):
            # d[a[i]] = b[i]
            d.update({a[i]: b[i]})
    return d

LR_Fed = pd.read_excel('E:\\work\\科研\\联邦学习\\心理\\结果整理\\LR_fed.xlsx')
LR_nonFed = pd.read_excel('E:\\work\\科研\\联邦学习\\心理\\结果整理\\LR_nonfed.xlsx')

RF_Fed = pd.read_excel('E:\\work\\科研\\联邦学习\\心理\\结果整理\\RF-fed.xlsx')
RF_nonFed = pd.read_excel('E:\\work\\科研\\联邦学习\\心理\\结果整理\\RF_nonfed.xlsx')

xgboost_Fed = pd.read_excel('E:\\work\\科研\\联邦学习\\心理\\结果整理\\xgboost_fed.xlsx')
xgboost_nonFed = pd.read_excel('E:\\work\\科研\\联邦学习\\心理\\结果整理\\xgboost_nonfed.xlsx')


df=LR_Fed.iloc[:,[0,2]]
tmp = df.to_dict(orient='list')#split
dic = convert2dic(tmp['sec'], tmp['auc'])
print(dic)

df=LR_nonFed.iloc[:,[0,2]]
tmp = df.to_dict(orient='list')#split
dic1 = convert2dic(tmp['sec'], tmp['auc'])
print(dic1)

df=RF_nonFed.iloc[:,[0,2]]
tmp = df.to_dict(orient='list')#split
print(tmp)
dic2 = convert2dic(tmp['sec'], tmp['auc'])
print(dic2)

df=RF_Fed.iloc[:,[0,2]]
tmp = df.to_dict(orient='list')#split
print(tmp)
dic3 = convert2dic(tmp['sec'], tmp['auc'])
print(dic3)

df=xgboost_Fed.iloc[:,[0,2]]
tmp = df.to_dict(orient='list')#split
print(tmp)
dic4 = convert2dic(tmp['sec'], tmp['auc'])
print(dic4)

df=xgboost_nonFed.iloc[:,[0,2]]
tmp = df.to_dict(orient='list')#split
print(tmp)
dic5 = convert2dic(tmp['sec'], tmp['auc'])
print(dic5)

#############plot
#results =[dic2, dic3]
#results =[dic4, dic5]
results =[dic, dic1]
print(results)
data_length = len(results[0])
# 将极坐标根据数据长度进行等分
angles = np.linspace(0, 2*np.pi, data_length, endpoint=False)
labels = [key for key in results[0].keys()]
score = [[v for v in result.values()] for result in results]
# 使雷达图数据封闭
score_a = np.concatenate((score[0], [score[0][0]]))
score_b = np.concatenate((score[1], [score[1][0]]))
'''score_c = np.concatenate((score[2], [score[2][0]]))
score_d = np.concatenate((score[3], [score[3][0]]))'''
angles = np.concatenate((angles, [angles[0]]))
labels = np.concatenate((labels, [labels[0]]))
# 设置图形的大小
fig = plt.figure(figsize=(8, 10), dpi=100)
# 新建一个子图
ax = plt.subplot(111, polar=True)
# 绘制雷达图
ax.plot(angles, score_a, color='b')
ax.plot(angles, score_b, color='g')
'''ax.plot(angles, score_c, color='red')
ax.plot(angles, score_d, color='yellow')'''
# 设置雷达图中每一项的标签显示
ax.set_thetagrids(angles*180/np.pi, labels)
# 设置雷达图的0度起始位置
ax.set_theta_zero_location('N')
# 设置雷达图的坐标刻度范围
ax.set_rlim(0, 1)
# 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
ax.set_rlabel_position(300)
ax.set_title("LR")
plt.legend(["non_federate_auc", "federate_auc"], loc='right')
#plt.savefig('E:\\work\\科研\\联邦学习\\心理\\结果整理\\RF.tif')
#plt.savefig('E:\\work\\科研\\联邦学习\\心理\\结果整理\\XGboost.tif')
plt.savefig('E:\\work\\科研\\联邦学习\\心理\\结果整理\\LR.tif')
plt.show()

