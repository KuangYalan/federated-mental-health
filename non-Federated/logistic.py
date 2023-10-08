import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt


df = pd.DataFrame([[0,0,0,0,0,0]], columns=['sec', 'acc', 'auc', 'precision', 'recall', 'f1'])
for sec in sections:
    row = []
    row.append(sec)

    best_acc = 0
    best_auc = 0
    train = pd.read_csv("D:/else/" + sec + "/train.csv", encoding='utf_8_sig')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_train = X_train.to_numpy()
    Y_train = np.reshape(y_train, [-1])

    test = pd.read_csv("D:/else/" + sec + "/test.csv", encoding='utf_8_sig')
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    X_test = X_test.to_numpy()
    Y_test = np.reshape(y_test, [-1])

    LR = linear_model.LogisticRegression(class_weight={0:0.2,1:0.8}, warm_start=True, max_iter=10000)
    LR.fit(X_train,Y_train)
    y_pre = LR.predict(X_test)

    fpr,tpr,thresholds = metrics.roc_curve(Y_test,y_pre)
    auc = metrics.auc(fpr,tpr)
    confusionmatrix = metrics.confusion_matrix(Y_test, y_pre)
    tn, fp, fn, tp = confusionmatrix.ravel()
    # print(confusionmatrix)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    acc = LR.score(X_test,Y_test)
    pre = metrics.precision_score(Y_test, y_pre)
    recall = metrics.recall_score(Y_test, y_pre)
    f1 = metrics.f1_score(Y_test, y_pre)

    print("acc:", acc)
    print("auc:", auc)
    print("sensitivity: ", sensitivity)
    print("pre:", pre)
    print("specificity: ", specificity)
    print("recall:", recall)
    print("F1 socre: ", f1)
    print("\n")
    row.append(acc)
    row.append(auc)
    row.append(pre)
    row.append(recall)
    row.append(f1)
    df.loc[len(df)] = row

    test = pd.read_csv("D:/else/" + sec + "/test.csv", encoding='utf_8_sig')
    X_test = test.iloc[:, :-1]
    explainer = shap.LinearExplainer(LR, X_test, feature_dependence="correlation")
    shap_values = explainer(X_test)
    plt.figure(figsize=(20, 10), dpi=100)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('D:/fig_LR/'+ sec + '_Feature_importance_LR.tif')
    df.to_excel('D:/else/LR_nonfed.xlsx', index=False)
