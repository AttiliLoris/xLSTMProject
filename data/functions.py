import numpy as np
import yaml


#funzione che manipola gli eventi implementando il one hot encoding e standardizzando il timestamp
def load_config(file_path):
    """Funzione per caricare il file YAML di configurazione."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def max_divisor(n, range):
    for d in range:
        if n % d == 0:
            return d
    return 1

def transform_vector3(array, max_time, row=100, col=12, lista_eventi=None):
    new_array = np.zeros((row, col), dtype=np.double)
    for i in range(len(array)):
        for k in range(len(lista_eventi)):
            if (array[i][0] == lista_eventi[k]):
                new_array[i][k] = 1
                break
        new_array[i][len(lista_eventi)] = array[i][1] / max_time
    return new_array

