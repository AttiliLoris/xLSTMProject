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
from nostriEsperimenti.funzioni import load_config, max_divisor

current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_script_dir)


def main():
    
    # # File di configurazione da terminale con --cfg
    # parser = argparse.ArgumentParser(description="Carica la configurazione del progetto")
    # parser.add_argument("--cfg", type=str, required=True, help="Percorso del file di configurazione YAML")
    # args = parser.parse_args()
    # config = load_config(args.cfg)
    config = load_config("config.yaml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DATAFILE= config['dataset']['path']
    X, Y, max_lenght, n_feature, activity_names= dataRead(DATAFILE)
    
    with open("nostriEsperimenti/output.txt", 'w') as f:
        #svuota il file di output prima di cominciare
        print('', file=f)
        
    DATAFILE= config['train']['batch_size']
    batch_sizes = config['train']['batch_size'] #Setting batch sizes
    kernel_size = config['train']['kernel_size'] #Setting kernel sizes
    lrs = config['train']['lr'] #Setting kernel sizes
    n_epochs = config['train']['epochs'] #Setting epochs
    
    for batch_size in batch_sizes:
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, shuffle=True)
        train_data = Data(X_train, Y_train)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        x_example = torch.zeros(batch_size, max_lenght, n_feature).to(device)
        depth = max_divisor(n_feature)
        clf = xlstm(['s','m','m','m','m','m'], x_example, max_lenght, depth= depth, factor=1).to(device)

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
                    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

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

                # Report e salvataggio su file
                labels = sorted(set(all_labels))
                with open("nostriEsperimenti/output.txt", 'a') as f:
                    print(correct, total, file=f)
                    print('depth: ',depth, file=f)
                    print(classification_report(all_labels, all_predictions, labels=labels, target_names=activity_names), file=f)
                    print(f'Batch size: {batch_size}, Epochs: {epochs}, lr: {lr}', file=f)
                    print('-' * 40, file=f)

            # Grafico per training e test loss
            print_plot(epochs, train_loss_values, test_loss_values, batch_size, lr, current_script_dir)
            

if __name__ == '__main__':
    main()

