import numpy.linalg as la
import math
import statistics

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import pandas as pd
from scipy.stats.stats import pearsonr


def evaluation(a, b):
    print(a, b)
    # a1 = np.array(a, dtype=np.float)
    # b1 = np.array(b, dtype=np.float)
    a1, b1 = a, b
    rmse = math.sqrt(mean_squared_error(a1,b1))
    mae = mean_absolute_error(a1, b1)
    a1 = a1.flatten()
    b1 = b1.flatten()
    if not statistics.stdev(a1) or not statistics.stdev(b1):
        cor = 1
    else:
        cor, _ = pearsonr(a1, b1)

    if not statistics.stdev(a1):
        r2 = 1
    else:
        r2 = r2_score(a1, b1)

    return rmse, mae, cor, r2


def process_per_segment(whos, test, result):
    answer =[[whos, whos, whos, whos]]
    answer.append(['RMSE', 'MAE', 'COR', 'R2'])
    print(test.shape, result.shape)
    for i in range(len(test)):
        rmse, mae, cor, r2 = evaluation(test[i], result[i])
        answer.append([rmse, mae, cor, r2])
    rmse, mae, cor, r2 = evaluation(test, result)
    answer.append([rmse, mae, cor, r2])

    return answer
