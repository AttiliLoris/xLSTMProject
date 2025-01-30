# xLSTMProject
Questo progetto mira a investigare le prestazioni di reti neurali xLSTM per il task di next activity prediction.
Per fare quanto sopra sono stati manipolati i dati dei log delle attività utilizzando i moduli di codice presenti nella cartella "data", successivamente i dati, diventati un insieme di prefissi, vengono utilizzati per addestrare una rete neurale xLSTM (https://github.com/styalai/xLSTM-pytorch.git) e i risultati vengono mostrati attraverso i grafici della funzione di loss e le statistiche nel file output.txt.

## Installazione 
Per installare il codice sorgente del progetto, basta eseguire il seguente codice dal terminale del proprio IDE:

```
git clone https://github.com/AttiliLoris/xLSTMProject
```

Una volta scaricato, si possono eseguire i seguenti comandi per scaricare le dipendenze descritte nel file requirementes.txt:

```
cd xLSTMProject
pip install -r requirements.txt
```

## Utilizzo
Per utilizzare il codice è sufficiente modificare il file [config.yaml](https://github.com/AttiliLoris/xLSTMProject/blob/main/config.yaml), inserendo nel campo "name" il nome del file contenente il dataset, scegliendo tra uno di quelli presenti nella cartella data/dataset, o, eventualmente, inserendone uno nuovo. Nel file config è anche possibile inserire i valori desiderati per i parametri della rete, che verranno poi utilizzati per effettuare una grid search, l'insieme di valori per un parametro deve essere elencato utilizzando la notazione vettoriale. Le metriche dell'addestramento e le matrici di confusione saranno consultabili nel file output.txt, mentre i grafici dell'andamento della funzione di loss verranno memorizzati nella cartella [loss_plot](https://github.com/AttiliLoris/xLSTMProject/tree/main/loss_plot).

