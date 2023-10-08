import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
import shap

df = pd.DataFrame([[0, 0,0,0,0,0]], columns=['sec', 'acc', 'auc', 'precision', 'recall', 'f1'])
for sec in sections:
    row = []
    row.append(sec)
    train = pd.read_csv("D:/else/"+ sec + "/train.csv",encoding='utf_8_sig')
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    X_train = X_train.to_numpy()
    Y_train = np.reshape(y_train,[-1])

    test = pd.read_csv("D:/else/"+ sec + "/test.csv",encoding='utf_8_sig')
    X_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1]
    X_test = X_test.to_numpy()
    Y_test = np.reshape(y_test,[-1])

    model = RandomForestClassifier(max_depth=8, n_estimators=7,random_state=2022)
    model.fit(X_train,Y_train)
    y_pre = model.predict(X_test)
    y_pre_prob = model.predict_proba(X_test)
    fpr,tpr,thresholds = metrics.roc_curve(Y_test,y_pre_prob[:,1])
    auc = metrics.auc(fpr,tpr)
    confusionmatrix = metrics.confusion_matrix(Y_test, y_pre)
    tn, fp, fn, tp = confusionmatrix.ravel()
    # print(confusionmatrix)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    acc = model.score(X_test,Y_test)
    recall = metrics.recall_score(Y_test, y_pre)
    precision = metrics.precision_score(Y_test, y_pre)
    f1 = metrics.f1_score(Y_test, y_pre)

    print("acc:", acc)
    print("auc:", auc)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1 socre: ", metrics.f1_score(Y_test, y_pre))
    print('\n')
    row.append(acc)
    row.append(auc)
    row.append(precision)
    row.append(recall)
    row.append(f1)
    df.loc[len(df)] = row

    test = pd.read_csv("D:/else/"+ sec + "/test.csv", encoding='utf_8_sig')
    X_test = test.iloc[:, :-1]

    df.loc[len(df)] = row
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    #print(shap_values)
    plt.figure(figsize=(20, 10), dpi=100)
    shap.summary_plot(shap_values[:,:,1], X_test, show=False)
    plt.savefig('D:/fig/' + sec + 'Feature_importance_RF.tif')
df.to_excel("D:/RF_sigle_client.xlsx")
