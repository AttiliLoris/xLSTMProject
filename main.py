from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import yaml
from xLSTM.xLSTM import xLSTM as xlstm
from sklearn.model_selection import train_test_split
from nostriEsperimenti.letturaDati import dataRead
from dataset import Data
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import argparse

# DATAFILE= 'C:/Users/loris/PycharmProjects/xLSTM-pytorch/examples/nostriEsperimenti/Helpdesk.xes'
DATAFILE= 'C:/Users/alep9/OneDrive/Desktop/UNIVERSITA/Big Data Analytics e Machine Learning/xLSTMProject/nostriEsperimenti/Helpdesk.xes'

def load_config(file_path):
    """Funzione per caricare il file YAML di configurazione."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
    
def main():
    
    # # File di configurazione da terminale con --cfg
    # parser = argparse.ArgumentParser(description="Carica la configurazione del progetto")
    # parser.add_argument("--cfg", type=str, required=True, help="Percorso del file di configurazione YAML")
    # args = parser.parse_args()
    # config = load_config(args.cfg)
    config = load_config("config.yaml")
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, max_lenght, n_feature = dataRead(DATAFILE)
    # with open("C:/Users/loris/PycharmProjects/xLSTM-pytorch/examples/nostriEsperimenti/output.txt", 'w') as f:
    with open("C:/Users/alep9/OneDrive/Desktop/UNIVERSITA/Big Data Analytics e Machine Learning/xLSTMProject/nostriEsperimenti/output.txt", 'w') as f:
        #svuota il file di output prima di cominciare
        print('', file=f)
    
    batch_sizes = config['train']['batch_size'] #Setting batch sizes
    kernel_size = config['train']['kernel_size'] #Setting kernel sizes
    lrs = config['train']['lr'] #Setting kernel sizes
    n_epochs = config['train']['epochs'] #Setting epochs
    
    for batch_size in batch_sizes:
        for lr in lrs:
            train_loss_values = []
            for epochs in [n_epochs]:

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, shuffle=True)
                train_data = Data(X_train, Y_train)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
                x_example = torch.zeros(batch_size, max_lenght, n_feature).to(device)
                clf = xlstm('m', x_example, kernel_size, factor=1).to(device)

                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(clf.parameters(), lr=lr)

                for epoch in range(epochs):
                    running_loss = 0.0
                    for i, data in enumerate(train_loader, 0):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        # set optimizer to zero grad to remove previous epoch gradients
                        optimizer.zero_grad()

                        # forward propagation
                        outputs = clf(inputs)
                        loss = criterion(outputs, labels.argmax(dim=1))
                        # backward propagation
                        loss.backward()
                        # optimize
                        optimizer.step()
                        running_loss += loss.item()
                    
                    # save train loss values
                    train_loss_value = running_loss / len(train_loader)
                    train_loss_values.append(train_loss_value)
                    
                    # display statistics
                    print(f'Epoch: {epoch + 1} - Numero di batch elaborati: {i + 1}  - Loss: {train_loss_value:.5f}')
                    
                PATH = './mymodel.pth'
                torch.save(clf.state_dict(), PATH)

                test_data = Data(X_test, Y_test)
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)


                correct, total = 0, 0
                all_labels = []
                all_predictions = []
                # no need to calculate gradients during inference
                with torch.no_grad():
                    for data in test_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        # calculate output by running through the network
                        outputs = clf(inputs)
                        # get the predictions
                        #__, predicted = torch.max(outputs.data, 1)
                        predicted = outputs.data.argmax(dim=1)
                        labels = labels.argmax(dim=1)
                        # update results
                        total += labels.size(0)
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        correct += (predicted == labels).sum()
                labels = sorted(set(all_labels)) # imposta le label per il report in modo che siano tutte e sole quelle contenute nei dati
                # with open("C:/Users/loris/PycharmProjects/xLSTM-pytorch/examples/nostriEsperimenti/output.txt",
                with open("C:/Users/alep9/OneDrive/Desktop/UNIVERSITA/Big Data Analytics e Machine Learning/xLSTMProject/nostriEsperimenti/output.txt",
                          'a') as f:
                    print(correct, total, file=f)
                    print(classification_report(all_labels, all_predictions, labels = labels, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8',
                                                                                           '9', '10']), file=f)
                    print(f'Batch size:{batch_size}, Epochs: {epochs}, lr: {lr}', file = f)
                    print('-' * 40, file=f)
            
            # Grafico loss
            plt.figure(figsize=(8, 5))
            plt.plot(list(range(1,epochs+1)), train_loss_values, marker='o', linestyle='-', color='b', label='Loss')
            plt.title('Loss durante le epoche')
            plt.xlabel('Epoche')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"C:/Users/alep9/OneDrive/Desktop/UNIVERSITA/Big Data Analytics e Machine Learning/xLSTMProject/loss_graphics/grafico_loss_{batch_size}_{lr}.png", format='png', dpi=300) 
            plt.close()  
if __name__ == '__main__':
    main()
