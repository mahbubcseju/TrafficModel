import numpy.linalg as la
import math
import statistics

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import pandas as pd
from scipy.stats.stats import pearsonr


def evaluation(a, b):
    a = np.array(a, dtype=np.float)
    b = np.array(b, dtype=np.float)
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    a = a.flatten()
    b = b.flatten()
    if not statistics.stdev(a) or not statistics.stdev(b):
        cor = 1
    else:
        cor, _ = pearsonr(a, b)

    if not statistics.stdev(a):
        r2 = 1
    else:
        r2 = r2_score(a, b)

    return rmse, mae, cor, r2


def process_per_segment(whos, test, result):
    answer =[[whos, whos, whos, whos]]
    answer.append(['RMSE', 'MAE', 'COR', 'R2'])

    for i in range(len(test)):
        rmse, mae, cor, r2 = evaluation(test[i], result[i])
        answer.append([rmse, mae, cor, r2])
    rmse, mae, cor, r2 = evaluation(test, result)
    answer.append([rmse, mae, cor, r2])

    return answer
