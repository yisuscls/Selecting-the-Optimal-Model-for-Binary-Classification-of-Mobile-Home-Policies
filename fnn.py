# FeedFordward Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import json
from imblearn.over_sampling import SMOTE 



#CREATE NEURAL NETWORK
class FNN(nn.Module):
    def __init__(self, hidden_size=85,hidden_layers=5):
        super(FNN, self).__init__()
        # Capa de entrada a capa oculta
         # Crear la lista de capas
        layers = []
        
        # Capa de entrada
        layers.append(nn.Linear(85, hidden_size))
        layers.append(nn.ReLU())

        
        # Crear capas ocultas dinámicamente
        for i in range(1, hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append( nn.Dropout(0.2))
            layers.append(nn.BatchNorm1d(hidden_size)) 
            layers.append(nn.ReLU())
            
            
        
        # Capa de salida
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())  # Activación sigmoide para clasificación binaria
        
        # Registrar las capas como un módulo secuencial
        self.network = nn.Sequential(*layers)
        pass

    def forward(self, x):
        return self.network(x)
    def show_net(self):
        device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )

        print(f"Using {device} device")
        print(FNN().to(device))
        pass

    pass
def train(fnn_model, features_tensor, labels_tensor, epochs=10, batch_size=1):
    loss_history = []
    class_counts = [5474, 348]
    weights = [ sum(class_counts)/c  for c in class_counts]
    criterion = nn.BCELoss()
    optimizer = optim.RAdam(fnn_model.parameters(), lr=0.001)

    for epoch in range(epochs):
        fnn_model.train()
        epoch_loss = 0

        for i in range(0, len(features_tensor), batch_size):
            batch_features = features_tensor[i:i + batch_size]
            batch_labels = labels_tensor[i:i + batch_size].float()

            outputs = fnn_model(batch_features).squeeze()
            w=  torch.tensor([weights[int(c)] for c in batch_labels], dtype=torch.float)
            criterion.weight=w
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fnn_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= (len(features_tensor) // batch_size)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.5f}')

    return loss_history


if __name__ == '__main__':

    #CREATE MODEL
    fnn_model=FNN(85,2)
    #torch.save(model.state_dict(), 'simple_nn.pth')
    print(fnn_model)
    
    #SELECT TRAIN DATA
    data=pd.read_csv('data/tic_2000_train_data.csv')
    #data=pd.read_csv('data/balanced_data.csv')
    features= data.iloc[:, :-1].values  #FEATURES
    labels = data.iloc[:, -1].values  #LABES  
   
    #smote = SMOTE(random_state=42,k_neighbors=10)
    #X_resampled, y_resampled = smote.fit_resample(features, labels)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    loss_history=train(fnn_model,features_tensor,labels_tensor,300,100)

    fnn_model.eval()  # Modo de evaluación
    with torch.no_grad():  # No calculamos gradientes en esta fase
        test_output = fnn_model(features_tensor).squeeze()
        predicted = (test_output > 0.5).float()  # Convertir probabilidades en predicciones 0 o 1

    # Calcular la precisión
    accuracy = (predicted == labels_tensor).float().mean()
    with open('index.json', 'r') as file:
        index = json.loads(file.read())
    index["fnn_index"]= index["fnn_index"]+1
    file_name=f'models/fnn/fnn_{index['fnn_index'] }_{int(accuracy*100)}.pth'
    torch.save(fnn_model, file_name)
    print(file_name)

    with open('index.json', 'w') as file:
        json.dump(index, file, indent=4)
    print(f'Accuracy: {accuracy.item():.4f}')
    
    plt.plot(loss_history)
    plt.title('Loss Function over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    pass