import argparse

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
import shap
import matplotlib.pyplot as plt

def model_validation_args_parser():
    parser = argparse.ArgumentParser(description="Validate model performance")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model file",
    )
    parser.add_argument(
        "--size_valid", type=int, help="Validation size, the first N instances to be treated as validation data"
    )
    parser.add_argument(
        "--num_trees",
        type=int,
        help="Total number of trees",
    )
    parser.add_argument(
        "--tree_method", type=str, default="hist", help="tree_method"
    )
    parser.add_argument(
        "--sec", type=str, default="test", help="sections"
    )
    return parser


def main():
    parser = model_validation_args_parser()
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    num_trees = args.num_trees
    sec = args.sec
    param = {}
    param["objective"] = "binary:logistic"
    param["eta"] = 0.1
    param["max_depth"] = 8
    param["eval_metric"] = "auc"
    param["nthread"] = 16
    param["num_parallel_tree"] = num_trees
    

    new_row = []
    new_row.append(sec)
    wb = pd.read_excel('./result.xlsx')
    # get validation data
    size_valid = args.size_valid
    #data = pd.read_csv(data_path, header=None, nrows=size_valid)
    data = pd.read_csv(data_path)
    #data = data.drop(0)
    # split to feature and label
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    dmat = xgb.DMatrix(X, label=y)

    # validate model performance
    bst = xgb.Booster(param, model_file=model_path)

    '''explainer = shap.TreeExplainer(bst)
    shap_values = explainer(X)
    plt.figure(figsize=(20,10),dpi=100)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('./fig/Feature_importance_RF_all.tif')'''

    #xgb.plot_importance

    y_pred = bst.predict(dmat)
    auc = roc_auc_score(y, y_pred)
    print(f"AUC over first {size_valid} instances is: {auc}")
    y_pred = np.array(y_pred)
    y_pred  = y_pred > 0.5
    y_pred = y_pred.astype(int)
    confusionmatrix = metrics.confusion_matrix(y, y_pred)
    tn, fp, fn, tp = confusionmatrix.ravel()
    # print(confusionmatrix)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    acc = metrics.accuracy_score(y, y_pred)
    pre = tp/(tp+fp)
    precision = tp / (tp + fp)
    recall = tp/(tp+fn)
    f1 = (2*pre*recall)/(pre+recall)
    print(" acc:", acc)
    print(" auc:", auc)
    print(" precision: ", precision)
    print(" recall: ", recall)
    print(" F1 socre: ", f1)
    new_row.append(acc)
    new_row.append(auc)
    new_row.append(precision)
    new_row.append(recall)
    new_row.append(f1)
    wb.loc[len(wb)] = new_row

    wb.to_excel("./result_RF.xlsx",index=False)


if __name__ == "__main__":
    main()=
