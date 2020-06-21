import numpy.linalg as la
import math

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd
from scipy.stats.stats import pearsonr


def evaluation(a, b):
    a = np.array(a, dtype=np.float)
    b = np.array(b, dtype=np.float)
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    a = a.flatten()
    b = b.flatten()
    cor, _ = pearsonr(a, b)
    return rmse, mae, cor


def process_per_segment(whos, test, result):
    answer =[[whos, whos, whos]]
    answer.append(['RMSE', 'MAE', 'COR'])

    for i in range(len(test)):
        rmse, mae, cor = evaluation(test[i], result[i])
        answer.append([rmse, mae, cor])
    rmse, mae, cor = evaluation(test, result)
    answer.append([rmse, mae, cor])

    return answer
