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


def transform_vector2(array, max_time, row=100, col=12):
    new_array = np.zeros((row, col))
    #getcontext().prec = 50
    for i in range(len(array)):
        match array[i][0]:
            case "START":
                new_array[i][0] = 1
            case "APARTLYSUBMITTED":
                new_array[i][1] = 1
            case "OSELECTED":
                new_array[i][2] = 1
            case "WWijzigencontractgegevens":
                new_array[i][3] = 1
            case "WNabellenincompletedossiers":
                new_array[i][4] = 1
            case "OACCEPTED":
                new_array[i][5] = 1
            case "AREGISTERED":
                new_array[i][6] = 1
            case "OSENT":
                new_array[i][7] = 1
            case "OSENTBACK":
                new_array[i][8] = 1
            case "OCANCELLED":
                new_array[i][9] = 1
            case "ADECLINED":
                new_array[i][10] = 1
            case "WAfhandelenleads":
                new_array[i][11] = 1
            case "WBeoordelenfraude":
                new_array[i][12] = 1
            case "OCREATED":
                new_array[i][13] = 1
            case "AFINALIZED":
                new_array[i][14] = 1
            case "AAPPROVED":
                new_array[i][15] = 1
            case "ACANCELLED":
                new_array[i][16] = 1
            case "APREACCEPTED":
                new_array[i][17] = 1
            case "WNabellenoffertes":
                new_array[i][18] = 1
            case "AACTIVATED":
                new_array[i][19] = 1
            case "ASUBMITTED":
                new_array[i][20] = 1
            case "WCompleterenaanvraag":
                new_array[i][21] = 1
            case "AACCEPTED":
                new_array[i][22] = 1
            case "ODECLINED":
                new_array[i][23] = 1
            case "WValiderenaanvraag":
                new_array[i][24] = 1
            case "END":
                new_array[i][25] = 1
            # case _:
            #     new_array
        #new_array[i][11] = Decimal(array[i][1] / max_time)
        new_array[i][26] = array[i][1]
    return new_array