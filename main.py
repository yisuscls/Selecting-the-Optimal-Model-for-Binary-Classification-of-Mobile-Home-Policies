import torch
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import pandas as pd
from fnn import FNN
from sklearn.svm import SVC
import joblib
import json

def fnn_eval(features_tensor, labels_tensor):
    model = torch.load("models/fnn/fnn_14_97.pth")
    model.eval()

    with torch.no_grad():
        test_output = model(features_tensor).squeeze()
        predicted = (test_output > 0.5).float()

    labels_np = labels_tensor.cpu().numpy()
    predicted_np = predicted.cpu().numpy()

    return {
        'Accuracy': accuracy_score(labels_np, predicted_np),
        'Recall': recall_score(labels_np, predicted_np, average='binary'),
        'Precision': precision_score(labels_np, predicted_np, average='binary'),
        'Confusion Matrix': confusion_matrix(labels_np, predicted_np).tolist()  # Convert matrix to list for display
    }

def model_eval(X, y, path):
    model = joblib.load(path)
    if 'tab_net' in path:
        # Convert DataFrame to numpy if the model is TabNet
        X = X.to_numpy().astype(float)
        y = y.to_numpy().astype(int)
        y_pred = model.predict(X)
    else:
        # Handle other models, assuming they can work directly with DataFrame inputs
        y_pred = model.predict(X)

    return {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='binary'),
        'Recall': recall_score(y, y_pred, average='binary'),
        'Confusion Matrix': confusion_matrix(y, y_pred).tolist()
    }




if __name__ == '__main__':
    data = pd.read_csv('data/tic_2000_train_data.csv')
    x_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]
    
    features_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    labels_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    x_test = pd.read_csv('data/tic_2000_eval_data.csv')
    y_test = pd.read_csv('data/tic_2000_target_data.csv')
    test_features_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    test_labels_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Evaluating models
    train_fnn_metrics = fnn_eval(features_tensor, labels_tensor)
    test_fnn_metrics = fnn_eval(test_features_tensor, test_labels_tensor)

    path_svm='models/svm/svm_rbf_17_ac_68_rc_83.pkl'
    train_svm_metrics = model_eval(x_train, y_train,path_svm)
    test_svm_metrics = model_eval(x_test, y_test,path_svm)

    path_nbm='models/nbm/nbm.pkl'
    train_nbm_metrics = model_eval(x_train, y_train,path_nbm)
    test_nbm_metrics = model_eval(x_test, y_test,path_nbm)

    path_lda='models/lda/lda.pkl'
    train_lda_metrics = model_eval(x_train, y_train,path_lda)
    test_lda_metrics = model_eval(x_test, y_test,path_lda)
    
    path_tabnet='models/tab_net/tab_net_2_ac_85_rc_11.pkl'
    train_tabnet_metrics = model_eval(x_train, y_train,path_tabnet)
    test_tabnet_metrics = model_eval(x_test, y_test,path_tabnet)


    # Dataframes for metrics
    metrics_train = pd.DataFrame({
        'Accuracy': [train_fnn_metrics['Accuracy'], train_svm_metrics['Accuracy'], train_nbm_metrics['Accuracy'], train_lda_metrics['Accuracy'], train_tabnet_metrics['Accuracy']],
        'Precision': [train_fnn_metrics['Precision'], train_svm_metrics['Precision'], train_nbm_metrics['Precision'], train_lda_metrics['Precision'], train_tabnet_metrics['Precision']],
        'Recall': [train_fnn_metrics['Recall'], train_svm_metrics['Recall'], train_nbm_metrics['Recall'], train_lda_metrics['Recall'], train_tabnet_metrics['Recall']],
        'Confusion Matrix': [train_fnn_metrics['Confusion Matrix'], train_svm_metrics['Confusion Matrix'], train_nbm_metrics['Confusion Matrix'], train_lda_metrics['Confusion Matrix'], train_tabnet_metrics['Confusion Matrix']]
    }, index=['FNN', 'SVM', 'NBM', 'LDA', 'TabNet'])

    metrics_test = pd.DataFrame({
        'Accuracy': [test_fnn_metrics['Accuracy'], test_svm_metrics['Accuracy'], test_nbm_metrics['Accuracy'], test_lda_metrics['Accuracy'], test_tabnet_metrics['Accuracy']],
        'Precision': [test_fnn_metrics['Precision'], test_svm_metrics['Precision'], test_nbm_metrics['Precision'], test_lda_metrics['Precision'], test_tabnet_metrics['Precision']],
        'Recall': [test_fnn_metrics['Recall'], test_svm_metrics['Recall'], test_nbm_metrics['Recall'], test_lda_metrics['Recall'], test_tabnet_metrics['Recall']],
        'Confusion Matrix': [test_fnn_metrics['Confusion Matrix'], test_svm_metrics['Confusion Matrix'], test_nbm_metrics['Confusion Matrix'], test_lda_metrics['Confusion Matrix'], test_tabnet_metrics['Confusion Matrix']]
    }, index=['FNN', 'SVM', 'NBM', 'LDA', 'TabNet'])

    print("Training Data Metrics:")
    print(metrics_train)
    print("\nTesting Data Metrics:")
    print(metrics_test)
