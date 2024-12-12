from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
import pandas as pd
import joblib
# Cargar datos de ejemplo (Iris dataset)
data=pd.read_csv('data/tic_2000_train_data.csv')
X_train= data.iloc[:, :-1].values  #FEATURES
y_train= data.iloc[:, -1].values  #LABES   


# Crear y entrenar el modelo LDA
lda = LinearDiscriminantAnalysis(solver='svd' ,priors=[0.55,0.45])

lda.fit(X_train, y_train)

# Realizar predicciones
y_pred = lda.predict(X_train)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_train, y_pred)

conf_matrix = confusion_matrix(y_train, y_pred)
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred, average='binary')  # Cambia a 'macro' si hay más de dos clases
recall = recall_score(y_train, y_pred, average='binary')

# Mostrar los resultados

print("Precisión (Accuracy):", accuracy)
print("Matriz de Confusión:\n", conf_matrix)
print("Precisión (Precision):", precision)
print("Recall:", recall)
joblib.dump(lda, f'models/lda/lda.pkl')