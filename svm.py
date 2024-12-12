# Support Vector Machine
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import joblib
import json
data=pd.read_csv('data/tic_2000_train_data.csv')
X= data.iloc[:, :-1].values  #FEATURES
y = data.iloc[:, -1].values  #LABES   


def best_kernel(X,y):

    # Estandarizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Definir los kernels a evaluar
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    # Inicializar variables para almacenar el mejor kernel y la mejor precisión
    best_accuracy = 0
    best_kernel = None

    # Probar cada kernel usando validación cruzada
    for k in kernels:
        svm = SVC(kernel=k)
        scores = cross_validate(svm, X_scaled, y, cv=10)
        accuracy = scores['test_score'].mean()  # Promedio de precisión de validación cruzada
        
        # Seleccionar el kernel con la mejor precisión
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_kernel = k
    
    print(f"Best Cross-Validation Accuracy: {best_accuracy:.2f}")
    return best_kernel
k= best_kernel(X,y)
for weight in range(1,18):
    print('#'*50)
    print('Weight :', weight)
    best_svm = SVC(kernel=k,class_weight={0: 1, 1: weight})
    best_svm.fit(X, y)

    # Hacer predicciones
    y_pred = best_svm.predict(X)

    # Calcular la matriz de confusión, precisión, recall y precisión general
    conf_matrix = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary')  # Cambia a 'macro' si hay más de dos clases
    recall = recall_score(y, y_pred, average='binary')
    with open('index.json', 'r') as file:
        index = json.loads(file.read())
    index['svm_index'] =index['svm_index'] + 1
    joblib.dump(best_svm, f'models/svm/svm_{k}_{index['svm_index'] }_ac_{int(accuracy*100)}_rc_{int(recall*100)}.pkl')

    with open('index.json', 'w') as file:
        json.dump(index, file, indent=4)
    # Mostrar los resultados
    print("Mejor Kernel:", k)
    print("Precisión (Accuracy):", accuracy)
    print("Matriz de Confusión:\n", conf_matrix)
    print("Precisión (Precision):", precision)
    print("Recall:", recall)
    print('#'*50)


