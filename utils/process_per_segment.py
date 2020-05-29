import numpy.linalg as la
import math

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd


def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    return rmse, mae


def process_per_segment(whos, test, result):
    answer =[[whos, whos]]
    answer.append(['RMSE', 'MAE'])

    for i in range(len(test)):
        rmse, mae = evaluation(test[i], result[i])
        answer.append([rmse, mae])
    rmse, mae = evaluation(test, result)
    answer.append([rmse, mae])

    return np.array(answer)
