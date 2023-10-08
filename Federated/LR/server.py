import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import numpy as np
from sklearn import metrics
import pickle
import shap
import matplotlib.pyplot as plt

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    #_, (X_test, y_test) = utils.load_mnist()
    test = pd.read_csv("data/test_all.csv",encoding='utf_8_sig')
    partition_id = np.random.choice(5)
    test_y0 = test.loc[(test['11']==0)]
    test_y1 = test.loc[(test['11']>0)]
    (test_y0, test_y1) = utils.partition(test_y0, test_y1, 1)[partition_id]
    X_test_y0 = test_y0.iloc[:,:-1]
    X_test_y1 = test_y1.iloc[:,:-1]
    X_test = pd.concat([X_test_y0,X_test_y1],axis=0)
    X_test = X_test.to_numpy()

    y_test_y0 = test_y0.iloc[:,-1]
    y_test_y1 = test_y1.iloc[:,-1]
    y_test = pd.concat([y_test_y0,y_test_y1],axis=0)
    y_test = np.reshape(y_test,[-1])

    print(X_test)
    print(y_test)
    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        y_pre = model.predict(X_test)
        proba = model.predict_proba(X_test)
        loss = log_loss(y_test, proba)
        accuracy = model.score(X_test, y_test)
        fpr,tpr,thresholds = metrics.roc_curve(y_test,proba[:,1])
        auc = metrics.auc(fpr,tpr)
        confusionmatrix = metrics.confusion_matrix(y_test, y_pre)
        tn, fp, fn, tp = confusionmatrix.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = metrics.f1_score(y_test, y_pre)
        precision = metrics.precision_score(y_test, y_pre)
        recall = metrics.recall_score(y_test, y_pre)
        print("server acc is:",accuracy)
        print("server auc is:",auc)
        print("server sen is:",sensitivity)
        print("server spec is:",specificity)
        print("server pre is:",precision)
        print("server recall is:",recall)
        print("server f1 is:",f1)
        
        with open('svm_Fed.pickle','wb') as f:
            pickle.dump(model,f)
        
        return loss, {"accuracy": accuracy}
    
    return evaluate
# Start Flower server of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)

    ####run after model trained to get the feature importance
    '''
    with open('LR_Fed.pickle', 'rb') as f1:
        model = pickle.load(f1)    

    for sec in sections:
        print(sec)
        test = pd.read_csv("data/" + sec + "/test_tmp.csv")

        X_test = test.iloc[:,1:]
        y_test = test.iloc[:,0]
        print(X_test)
        explainer = shap.LinearExplainer(model, X_test, feature_dependence="correlation")
        shap_values = explainer(X_test)
        plt.figure(figsize=(20,10),dpi=100)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig('./fig/' + sec + 'Feature_importance_LR.tif')'''

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=10,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=50),
    )
