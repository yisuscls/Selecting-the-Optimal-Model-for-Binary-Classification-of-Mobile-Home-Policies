# Import necessary libraries
import pandas as pd

from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
import joblib 


df=pd.read_csv('data/tic_2000_train_data.csv')


# Step 2: Data Preprocessing
# Example: Handling categorical data using Label Encoding
# Adjust this based on your dataset's structure


# Step 3: Feature Selection and Splitting
# Assuming the last column is the target variable
X_train = df.iloc[:, :-1]  # Features
y_train = df.iloc[:, -1]   # Target variable
p=0.75

# Step 4: Train the Naive Bayes Model
model = BernoulliNB(class_prior=[p,1-p])
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
conf_matrix = confusion_matrix(y_train, y_pred)

precision = precision_score(y_train, y_pred, average='binary')  # Cambia a 'macro' si hay más de dos clases
recall = recall_score(y_train, y_pred, average='binary')


print("Accuracy:", accuracy)

print("Matriz de Confusión:\n", conf_matrix)
print("Precisión (Precision):", precision)
print("Recall:", recall)


joblib.dump(model, f'models/nbm/nbm.pkl')


