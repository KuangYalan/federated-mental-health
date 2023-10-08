import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier
import xgboost
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc
import pickle
import matplotlib.pyplot as plt
from xgboost import plot_importance
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


    model = XGBClassifier(max_depth=15, subsample=1, seed=42)
    eval_set = [(X_test, Y_test)]
    model.fit(X_train, Y_train, eval_set=eval_set, verbose=False)  # eval_metric="logloss",
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba[:, 1])
    acc = accuracy_score(Y_test, y_pred)
    myauc = auc(fpr, tpr)
    confusionmatrix = metrics.confusion_matrix(Y_test, y_pred)
    tn, fp, fn, tp = confusionmatrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    recall = metrics.recall_score(Y_test, y_pred)
    precision = tp / (tp + fp)
    f1 = metrics.f1_score(Y_test, y_pred)
    print(sec + "acc:", acc)
    print(sec +"auc:", myauc)
    print(sec +"pre: ", precision)
    print(sec +"recall:", recall)
    print(sec +"F1 socre: ", metrics.f1_score(Y_test, y_pred))
    print('\n')
    row.append(acc)
    row.append(myauc)
    row.append(precision)
    row.append(recall)
    row.append(f1)
    df.loc[len(df)] = row

    test = pd.read_csv("D:/else/"+ sec + "/test.csv",encoding='utf_8_sig')
    X_test = test.iloc[:, :-1]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    df.loc[len(df)] = row
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    plt.figure(figsize=(8, 6), dpi=100)
    shap.summary_plot(shap_values[:, :, 1], X_test, show=False)
    plt.savefig('D:/fig/' + sec + 'Feature_importance_XGBoost.tif')