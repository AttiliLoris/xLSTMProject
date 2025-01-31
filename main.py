import torch
import torch.nn as nn
from utils import print_plot
from xLSTM.xLSTM import xLSTM as xlstm
from sklearn.model_selection import train_test_split
from data.data_read import dataRead
from dataset import Data
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
import os
import sys
from data.functions import load_config, max_divisor

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
    
    DATAFILE= "data/dataset/" + config['dataset']['name']
    X, Y, max_lenght, n_feature, activity_names = dataRead(DATAFILE)


    with open("output.txt", 'w') as f:
        # Svuota il file di output prima di cominciare
        print('', file=f)

    batch_sizes = config['train']['batch_size'] # Setting batch sizes
    lrs = config['train']['lr'] # Setting kernel sizes
    n_epochs = config['train']['epochs'] # Setting epochs
    layers_set_list = config['train']['layers_set_list'] # Setting layers
    depth_range = config['train']['depth_range'] # Setting depth size
    indice_test = 0
    for layers_set in layers_set_list:
        for batch_size in batch_sizes:

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, shuffle=True)
            train_data = Data(X_train, Y_train)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
            x_example = torch.zeros(batch_size, max_lenght, n_feature).to(device)
            depth = max_divisor(n_feature, depth_range)


            for lr in lrs:
                indice_test += 1
                clf = xlstm(layers_set, x_example, max_lenght, depth=depth, factor=1).to(device)
                train_loss_values = []
                test_loss_values = []

                for epochs in [n_epochs]:
                    patience = 10  # Numero massimo di epoche senza miglioramenti
                    best_loss = float('inf')  # Inizializza la migliore loss con un valore molto alto
                    early_stop_counter = 0  # Contatore delle epoche senza miglioramento

                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(clf.parameters(), lr=lr)
                    for epoch in range(epochs):
                        time.sleep(5)
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
                        test_loss = 0.0  # Aggiunge accumulatore per la test loss
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

                        if test_loss_value < best_loss:
                            best_loss = test_loss_value  # Aggiorna la migliore loss
                            early_stop_counter = 0  # Resetta il contatore
                        else:
                            early_stop_counter += 1  # Incrementa il contatore

                        if early_stop_counter >= patience:
                            epoch = epoch+1
                            print(f"Early stopping at epoch {epoch}. Best Test Loss: {best_loss:.5f}")
                            break


                    # Report e salvataggio su file
                    labels = sorted(set(all_labels))
                    cm = confusion_matrix(all_labels, all_predictions, labels=labels)

                    with open("output.txt", 'a') as f:
                        print("Test numero ", indice_test, file=f)
                        print('Correct: ', correct,'|  Total: ', total, file=f)
                        print(classification_report(all_labels, all_predictions, labels=labels, target_names=activity_names[1:]), file=f)
                        print(cm, file=f)
                        print(f'Batch size: {batch_size}, Epochs: {epoch}, lr: {lr}, Layers: {layers_set}, Depth: {depth}', file=f)
                        print('-' * 40, file=f)

                # Grafico per training e test loss
                print_plot(epoch, train_loss_values, test_loss_values, batch_size, lr, current_script_dir, layers_set, depth)


if __name__ == '__main__':
    main()

