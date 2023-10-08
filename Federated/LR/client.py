import warnings
import flwr as fl
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn import metrics,svm

import utils

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', type=str, default = None)
    args = parser.parse_args()
    print(args.section)
    train = pd.read_csv("data/" + args.section + "/train.csv",encoding='utf_8_sig')

    test = pd.read_csv("data/" + args.section + "/test.csv",encoding='utf_8_sig')
    X_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1]
    X_test = X_test.to_numpy()
    y_test = np.reshape(y_test,[-1])

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(5)
    train_y0 = train.loc[(train['11']==0)]
    train_y1 = train.loc[(train['11']>0)]
    (train_y0, train_y1) = utils.partition(train_y0, train_y1, 5)[partition_id]
    X_train_y0 = train_y0.iloc[:,:-1]
    X_train_y1 = train_y1.iloc[:,:-1]
    X_train = pd.concat([X_train_y0,X_train_y1],axis=0)
    X_train = X_train.to_numpy()
    
    y_train_y0 = train_y0.iloc[:,-1]
    y_train_y1 = train_y1.iloc[:,-1]
    y_train = pd.concat([y_train_y0,y_train_y1],axis=0)
    y_train = np.reshape(y_train,[-1])
    #print(train)
    #print(X_train)
    #print(y_train)
    # Create LogisticRegression Model
    model = LogisticRegression(
        #penalty="l2",
        max_iter=10000,  # local epoch
        class_weight={0:0.2,1:0.8},
        warm_start=True,  # prevent refreshing weights when fitting
    )
    #model = svm.SVC(kernel='linear', probability=True, max_iter=100)
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            y_pre = model.predict(X_test)
            proba = model.predict_proba(X_test)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            fpr,tpr,thresholds = metrics.roc_curve(y_test,proba[:,1])
            auc = metrics.auc(fpr,tpr)
            confusionmatrix = metrics.confusion_matrix(y_test, y_pre)
            tn, fp, fn, tp = confusionmatrix.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            f1 = metrics.f1_score(y_test, y_pre)
            pre = metrics.precision_score(y_test, y_pre)
            recall = metrics.recall_score(y_test, y_pre)
            print("client" + args.section + " acc is:",accuracy)
            print("client" + args.section + " auc is:",auc)
            print("client" + args.section + " sen is:",sensitivity)
            print("client" + args.section + " spec is:",specificity)
            print("client" + args.section + " pre is:",pre)
            print("client" + args.section + " recall is:",recall)
            print("client" + args.section + " f1 is:",f1)
            df = pd.read_excel('./LR_fed.xlsx')
            row = [args.section]
            row.append(accuracy)
            row.append(auc)
            row.append(pre)
            row.append(recall)
            row.append(f1)
            df.loc[len(df)] = row
            df.to_excel('./LR_fed.xlsx', index=False)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
