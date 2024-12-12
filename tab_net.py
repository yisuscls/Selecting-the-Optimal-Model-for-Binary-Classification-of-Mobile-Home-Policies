# Tab Net
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import joblib
import json
data=pd.read_csv('data/tic_2000_train_data.csv')
#data=pd.read_csv('data/balanced_data.csv')
X_train= data.iloc[:, :-1].values  #FEATURES
y_train = data.iloc[:, -1].values  #LABES

tabnet = TabNetClassifier(optimizer_fn=torch.optim.RAdam,
                          optimizer_params=dict(lr=1e-3),
                          scheduler_params={"step_size":50,"gamma":0.9},
                          scheduler_fn=torch.optim.lr_scheduler.StepLR,
                          mask_type='entmax'
                          )

tabnet.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train)],
    eval_name=['train'],
    eval_metric=['accuracy',],
    max_epochs=100,
    patience=20,
    batch_size=100, 
    virtual_batch_size=50,
    num_workers=0,
    drop_last=False
)
y_pred = tabnet.predict(X_train)

# Calcular la precisión
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred, average='binary')
recall = recall_score(y_train, y_pred, average='binary')
f1 = f1_score(y_train, y_pred, average='binary')
confusion= confusion_matrix(y_train, y_pred)
# Mostrar los resultados
print(f"Confusion Matrix {confusion}")
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

with open('index.json', 'r') as file:
    index = json.loads(file.read())
    index['tab_net_index'] =index['tab_net_index'] + 1
    joblib.dump(tabnet, f'models/tab_net/tab_net_{index['tab_net_index'] }_ac_{int(accuracy*100)}_rc_{int(recall*100)}.pkl')

with open('index.json', 'w') as file:
    json.dump(index, file, indent=4)