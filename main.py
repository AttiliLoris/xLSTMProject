from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import yaml
from utils import print_plot
from xLSTM.xLSTM import xLSTM as xlstm
from sklearn.model_selection import train_test_split
from nostriEsperimenti.letturaDati import dataRead
from dataset import Data
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import argparse
import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_script_dir)
DATAFILE= 'nostriEsperimenti/BPI12.xes'

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
    
    with open("nostriEsperimenti/output.txt", 'w') as f:
        #svuota il file di output prima di cominciare
        print('', file=f)
    
    batch_sizes = config['train']['batch_size'] #Setting batch sizes
    kernel_size = config['train']['kernel_size'] #Setting kernel sizes
    lrs = config['train']['lr'] #Setting kernel sizes
    n_epochs = config['train']['epochs'] #Setting epochs
    
    for batch_size in batch_sizes:
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, shuffle=False)
        train_data = Data(X_train, Y_train)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
        x_example = torch.zeros(batch_size, max_lenght, n_feature).to(device)
        clf = xlstm('m', x_example, max_lenght, factor=1).to(device)

        for lr in lrs:
            
            train_loss_values = []
            test_loss_values = []
            
            for epochs in [n_epochs]:

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
                    
                    test_data = Data(X_test, Y_test)
                    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

                    # Test loss e accuratezza
                    test_loss = 0.0  # Aggiungi accumulatore per la test loss
                    correct, total = 0, 0
                    all_labels = []
                    all_predictions = []

                    with torch.no_grad():
                        for data in test_loader:
                            inputs, labels = data
                            inputs, labels = inputs.to(device), labels.to(device)
                            
                            # Calcola output
                            outputs = clf(inputs)
                            
                            # Calcola la loss sul test set
                            loss = criterion(outputs, labels.argmax(dim=1))
                            test_loss += loss.item()
                            
                            # Ottieni le predizioni
                            predicted = outputs.data.argmax(dim=1)
                            labels = labels.argmax(dim=1)
                            
                            # Aggiorna risultati
                            total += labels.size(0)
                            all_predictions.extend(predicted.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                            correct += (predicted == labels).sum().item()

                        # Calcola la loss media del test
                        test_loss_value = test_loss / len(test_loader)
                        test_loss_values.append(test_loss_value)
                        
                        
                        # display statistics
                        print(f'Epoch: {epoch + 1} - Number of batches processed: {i + 1}  - Train Loss: {train_loss_value:.5f}, Test Loss: {test_loss_value:.5f}')

                PATH = './mymodel.pth'
                torch.save(clf.state_dict(), PATH)
                
                # Report e salvataggio su file
                labels = sorted(set(all_labels))
                with open("nostriEsperimenti/output.txt", 'a') as f:
                    print(f"Test Loss: {test_loss_value:.5f}", file=f)
                    print(correct, total, file=f)
                    print(classification_report(all_labels, all_predictions, labels=labels, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11','12',
                                                                                                          '13','14','15','16','17','18','19','20','21','22','23','24','25','26']), file=f)
                    print(f'Batch size: {batch_size}, Epochs: {epochs}, lr: {lr}', file=f)
                    print('-' * 40, file=f)

            # Grafico per training e test loss
            print_plot(epochs, train_loss_values, test_loss_values, batch_size, lr, current_script_dir)
            
            # plt.figure(figsize=(8, 5))
            # plt.plot(list(range(1, epochs + 1)), train_loss_values, marker='o', linestyle='-', color='b', label='Train Loss')
            # plt.plot(list(range(1, epochs + 1)), test_loss_values, marker='o', linestyle='--', color='r', label='Test Loss')
            # plt.title('Loss durante le epoche')
            # plt.xlabel('Epoche')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.grid(True)
            # plt.savefig(f"{current_script_dir}/loss_plot/grafico_loss_{batch_size}_{lr}.png", format='png', dpi=300)
            # plt.close()

if __name__ == '__main__':
    main()
