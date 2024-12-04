import torch
import torch.nn as nn
from xLSTM.xLSTM import xLSTM as xlstm
from sklearn.model_selection import train_test_split
from nostriEsperimenti.letturaDati import dataRead
from dataset import Data
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
DATAFILE= 'C:/Users/loris/PycharmProjects/xLSTM-pytorch/examples/nostriEsperimenti/Helpdesk.xes'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, max_lenght, n_feature = dataRead(DATAFILE)
    with open("C:/Users/loris/PycharmProjects/xLSTM-pytorch/examples/nostriEsperimenti/output.txt", 'w') as f:
        #svuota il file di output prima di cominciare
        print('', file=f)
    for batch_size in [32,64]:
        for lr in [0.01, 0.001, 0.1]:
            for epochs in [30]:

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, shuffle=True)
                train_data = Data(X_train, Y_train)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
                x_example = torch.zeros(batch_size, max_lenght, n_feature).to(device)
                clf = xlstm('m', x_example, factor=1).to(device)

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
                  # display statistics
                  print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(train_loader):.5f}')
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
                labels = sorted(set(all_labels)) # imposta le lables per il report in modo che siano tutte e sole quelle contenute nei dati
                with open("C:/Users/loris/PycharmProjects/xLSTM-pytorch/examples/nostriEsperimenti/output.txt",
                          'a') as f:
                    print(correct, total, file=f)
                    print(classification_report(all_labels, all_predictions, labels = labels, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8',
                                                                                           '9', '10']), file=f)
                    print(batch_size, epochs, lr, file = f)
                    print('-' * 40, file=f)
if __name__ == '__main__':
    main()
