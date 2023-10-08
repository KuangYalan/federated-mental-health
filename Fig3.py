import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

matplotlib.rcParams['font.family']='SimHei'     #set the typeface
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

def convert2dic(a,b):
    d = { }
    if len(a) == len(b):
        for i in range(len(a)):
            # d[a[i]] = b[i]
            d.update({a[i]: b[i]})
    return d

LR_Fed = pd.read_excel('E:\\work\\LR_fed.xlsx')
LR_nonFed = pd.read_excel('E:\\work\\LR_nonfed.xlsx')

RF_Fed = pd.read_excel('E:\\work\\RF-fed.xlsx')
RF_nonFed = pd.read_excel('E:\\work\\RF_nonfed.xlsx')

xgboost_Fed = pd.read_excel('E:\\work\\xgboost_fed.xlsx')
xgboost_nonFed = pd.read_excel('E:\\work\\xgboost_nonfed.xlsx')


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

angles = np.linspace(0, 2*np.pi, data_length, endpoint=False)
labels = [key for key in results[0].keys()]
score = [[v for v in result.values()] for result in results]

score_a = np.concatenate((score[0], [score[0][0]]))
score_b = np.concatenate((score[1], [score[1][0]]))
'''score_c = np.concatenate((score[2], [score[2][0]]))
score_d = np.concatenate((score[3], [score[3][0]]))'''
angles = np.concatenate((angles, [angles[0]]))
labels = np.concatenate((labels, [labels[0]]))

fig = plt.figure(figsize=(8, 10), dpi=100)

ax = plt.subplot(111, polar=True)

ax.plot(angles, score_a, color='b')
ax.plot(angles, score_b, color='g')
'''ax.plot(angles, score_c, color='red')
ax.plot(angles, score_d, color='yellow')'''

ax.set_thetagrids(angles*180/np.pi, labels)

ax.set_theta_zero_location('N')

ax.set_rlim(0, 1)

ax.set_rlabel_position(300)
ax.set_title("LR")
plt.legend(["non_federate_auc", "federate_auc"], loc='right')
#plt.savefig('E:\\work\\RF.tif')
#plt.savefig('E:\\work\\XGboost.tif')
plt.savefig('E:\\work\\LR.tif')
plt.show()

