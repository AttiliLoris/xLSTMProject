from typing import Optional

import pandas
import pm4py
import numpy as np
import xml.etree.ElementTree as ET
from .funzioni import transform_vector2
from .funzioni import transform_vector3


def dataRead(datafile):
    # Leggere un file .xes
    log = pm4py.read_xes(datafile)
    np.set_printoptions(threshold=np.inf)
    # il log è strutturato come vettore di tracce che sono vettori di eventi che sono vettori di features
    # ESPERIMENTO FUTURO, PER ORA SOLO INDICE ATTIVITà E TIMESTAMP
    # log = pm4py.project_on_event_attribute(log, ["concept:name","time:timestamp","case:concept:name","case:variant-index"])

    # converte il log in dataframe per utilizzare i metodi del dataframe
    df = pm4py.convert_to_dataframe(log)

    # Ottieni i valori distinti di concept:name
    unique_values = df["concept:name"].unique()

    # Converti in una lista se necessario
    lista_nomi_eventi = unique_values.tolist()
    n_feature = len(lista_nomi_eventi) +1  # +1 per comprendere START, END e timestamp


    # Trova il massimo e il minimo della colonna time:timestamp
    max_time = df['time:timestamp'].max().to_pydatetime().timestamp()
    min_time = df['time:timestamp'].min().to_pydatetime().timestamp()
    # AAAA numeric_values = pandas.to_numeric(df['concept:name'], errors='coerce')

    # Calcola il massimo ignorando i NaN (che corrispondono a valori non numerici)
    #AAAAAn_feature = int(numeric_values.max()) + 3
    
    # calcola il numero di feature dai metadati
    tree = ET.parse(datafile)
    root = tree.getroot()
    named_events_total = root.find(".//int[@key='meta_concept:named_events_total']")

    # denominatore della standardizzazione
    standard_time = max_time - min_time

    # proiezione del log sull'identificatore dell'attività e il timestamp
    log = pm4py.project_on_event_attribute(log, ["concept:name", "time:timestamp"])
    # calcolo lunghezza massima delle tracce e numero prefissi da generare
    trace_max_length = 0
    prefix_num = 0

    for i in range(len(log)):
        if len(log[i]) > trace_max_length:
            trace_max_length = len(log[i])
        prefix_num += len(log[i]) - 1
    
    

    # conversione dei timestamp in interi
    for i in range(len(log)):
        for j in range(len(log[i])):
            for k in range(len(log[i][j])):
                if isinstance(log[i][j][k], pandas.Timestamp):
                    log[i][j][k] = log[i][j][k].to_pydatetime().timestamp() - min_time




    # trasformazione degli eventi con onehot encoding e stardandizzazione del timestamp
    for i in range(len(log)):
        #log[i] = transform_vector2(log[i], standard_time, trace_max_length, n_feature)
        log[i] = transform_vector3(log[i], standard_time, trace_max_length, n_feature, lista_nomi_eventi)

    # creazione dei vettori di prefissi e maschere di predizione
    res = np.zeros((prefix_num, trace_max_length, n_feature)).astype(int)
    prediction_mask = np.zeros((prefix_num, n_feature-1)).astype(int)
    index = 0
    for i in range(len(log)):
        for j in range(trace_max_length - 1, 0, -1):  # cicla fino a rimuovere il primo elemento(escluso, i prefissi minimi hanno lo start)
            if all(x == 0 for x in log[i][j]):
                continue
            y = log[i][j].copy()
            log[i][j] = np.zeros(n_feature)
            res[index] = log[i]
            y = np.delete(y, -1)
            prediction_mask[index] = y
            index += 1
    return res, prediction_mask, trace_max_length, n_feature


''' analizzando le righe possiamo vedere che alcune colonne non sono utili per ML:
- org:resource
- lifecicle:transition (perché sono tutte complete) DA VERIFICARE
- creator
- case:concept:name????
- case:variant rimuovere perchè ridondante e scomodo

quindi rimangono:
-concept:name sostituendo le righe che hanno i valori "START" e "END" con altri indici numerici
- timestamp
-case:concept:name per evidenziare l'importanza delle attività precedenti che si legano (cioè l'importanza
di una determinata sequenza di eventi nella stessa traccia)
- case:variant-index????

FARE GROUPBY SUL LOG PER TROVARE IL MASSIMO NUMERODI EVENTI IN UNA TRACCIA, NECESSARIO PER DETERMINARE LA DIMENSIONE
DEI TENSORI DA DARE IN INPUT ALLA RETE
'''