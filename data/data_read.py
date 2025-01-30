import pandas
import pm4py
import numpy as np
from .functions import transform_vector


def dataRead(datafile):
    # Leggere un file .xes
    log = pm4py.read_xes(datafile)
    np.set_printoptions(threshold=np.inf)
    # il log è strutturato come vettore di tracce che sono vettori di eventi che sono vettori di features

    # converte il log in dataframe per utilizzare i metodi del dataframe
    df = pm4py.convert_to_dataframe(log)

    # Ottieni i valori distinti di concept:name
    unique_values = df["concept:name"].unique()

    # Converti in una lista
    activity_names = unique_values.tolist()
    n_feature = len(activity_names) +1  # +1 per comprendere START, END e timestamp


    # Trova il massimo e il minimo della colonna time:timestamp
    max_time = df['time:timestamp'].max().to_pydatetime().timestamp()
    min_time = df['time:timestamp'].min().to_pydatetime().timestamp()
    # AAAA numeric_values = pandas.to_numeric(df['concept:name'], errors='coerce')

    # Calcola il massimo ignorando i NaN (che corrispondono a valori non numerici)
    #AAAAAn_feature = int(numeric_values.max()) + 3
    
    

    # denominatore della standardizzazione
    standard_time = max_time - min_time

    # proiezione del log sull'identificatore dell'attività e il timestamp
    log = pm4py.project_on_event_attribute(log, ["concept:name", "time:timestamp"])
    # calcolo lunghezza massima delle tracce e numero prefissi da generare
    trace_max_length = 0
    prefix_num = 0

    #calcola la lunghezza massima di una traccia e il numero di prefissi che potranno essere generati
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
        log[i] = transform_vector(log[i], standard_time, trace_max_length, n_feature, activity_names)

    # creazione dei vettori di prefissi e maschere di predizione
    res = np.zeros((prefix_num, trace_max_length, n_feature), dtype=np.double)
    prediction_mask = np.zeros((prefix_num, n_feature-1), dtype=np.double)
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
    return res, prediction_mask, trace_max_length, n_feature, activity_names


