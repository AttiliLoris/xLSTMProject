from decimal import Decimal, getcontext

import numpy as np


#funzione che manipola gli eventi implementando il one hot encoding e standardizzando il timestamp
def transform_vector(array, max_time, row=100, col=12):
    new_array = np.zeros((row, col))
    #getcontext().prec = 50
    for i in range(len(array)):
        match array[i][0]:
            case "START":
                new_array[i][0] = 1
            case "1":
                new_array[i][1] = 1
            case "2":
                new_array[i][2] = 1
            case "3":
                new_array[i][3] = 1
            case "4":
                new_array[i][4] = 1
            case "5":
                new_array[i][5] = 1
            case "6":
                new_array[i][6] = 1
            case "7":
                new_array[i][7] = 1
            case "8":
                new_array[i][8] = 1
            case "9":
                new_array[i][9] = 1
            case "END":
                new_array[i][10] = 1
            # case _:
            #     new_array
        #new_array[i][11] = Decimal(array[i][1] / max_time)
        new_array[i][11] = array[i][1]
    return new_array


def mask_next_activity(array, res, prediction_mask):
    index = len(array)
    for i in range(index):
        elem = array[-1]
        array = array[:-1]
        res.append(array)
        prediction_mask.append(elem)
    return res, prediction_mask
