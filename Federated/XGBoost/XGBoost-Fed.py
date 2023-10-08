import argparse
import os
import tempfile
import time

import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np
import shap
import matplotlib.pyplot as plt


def get_training_parameters():
    # use logistic regression loss for binary classification
    # use auc as metric
    param = {
        "objective": "binary:logistic",
        "eta": 0.1,
        "max_depth": 8,
        "eval_metric": "auc",
        "nthread": 16,
    }
    return param


def main():
    # setup parameters for xgboost
    xgb_params = get_training_parameters()

    # test model
    model_path = "../tree-based/workspaces/simulate_job/app_server/xgboost_model.json"
    bst = xgb.Booster(xgb_params, model_file=model_path)
    
    train_X_column_name = ['Marital quality of parents','Social connection','Sex',' Student burnout','Grit','Stress', 'Internet addiction','Bullying level','Bullying participation', 'Positive coping', 'Negative coping']

    test = pd.read_csv("Fed-XGBoost/data/test_all.csv")
    print(test)

    sorted_idx = bst.feature_importances_.argsort()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(20,10),dpi=100)

    plt.barh(np.array(train_X_column_name)[sorted_idx], bst.feature_importances_[sorted_idx])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=23,rotation=20)#fontsize=5,
    plt.xlabel("Feature Importance",fontsize=23)
    plt.tight_layout()
    plt.savefig('./Feature_importance_all.tif')


    df = pd.DataFrame([[0, 0,0,0,0,0]], columns=['sec', 'acc', 'auc', 'precision', 'recall', 'f1'])
    for sec in sections:
        print(sec)
        row = [sec]
        test = pd.read_csv("Fed-XGBoost/data/" + sec + "/test.csv")
        X_test = test.iloc[:,1:]
        y_test = test.iloc[:,0]
        explainer = shap.TreeExplainer(bst, feature_names=train_X_column_name)
        shap_values = explainer(X_test)
        plt.figure(figsize=(20,10),dpi=100)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig('./fig/' + sec + '_Feature_importance.tif')

        test = xgb.DMatrix(X_test,label=y_test)
        y_pred = bst.predict(test)
        y_test = np.reshape(y_test,[-1])

        print(y_test)
        print(y_pred)
        fpr,tpr,thresholds = metrics.roc_curve(y_test[1:],y_pred[1:])
        auc = metrics.auc(fpr,tpr)
        y_pred = np.array(y_pred)
        y_pred  = y_pred > 0.5  
        y_pred = y_pred.astype(int)
        confusionmatrix = metrics.confusion_matrix(y_test[1:], y_pred[1:])
        tn, fp, fn, tp = confusionmatrix.ravel()
        # print(confusionmatrix)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        acc = metrics.accuracy_score(y_test[1:], y_pred[1:])
        pre = tp/(tp+fp)
        precision = tp / (tp + fp)
        recall = tp/(tp+fn)
        f1 = (2*pre*recall)/(pre+recall)
        print(sec + " acc:", acc)
        print(sec + " auc:", auc)
        print(sec + " precision: ", precision)
        print(sec + " recall: ", recall)
        print(sec + " F1 socre: ", f1)
        row.append(acc)
        row.append(auc)
        row.append(precision)
        row.append(recall)
        row.append(f1)
        df.loc[len(df)] = row
    df.to_excel('./xgboost_fed.xlsx',index=False)

if __name__ == "__main__":
    main()
