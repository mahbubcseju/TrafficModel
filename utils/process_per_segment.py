import numpy.linalg as la
import math
import statistics
from collections import Iterable

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import pandas as pd
from scipy.stats.stats import pearsonr


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def evaluation(a, b):
    # print(a, b)
    try:
        a1, b1 = list(flatten(a)), list(flatten(b))

        a1 = np.array(a1, dtype="float")
        b1 = np.array(b1, dtype="float")

        rmse = math.sqrt(mean_squared_error(a1,b1))
        mae = mean_absolute_error(a1, b1)

        if not statistics.stdev(a1) or not statistics.stdev(b1):
            cor = 1
        else:
            cor, _ = pearsonr(a1, b1)

        if not statistics.stdev(a1):
            r2 = 1
        else:
            r2 = r2_score(a1, b1)

        return rmse, mae, cor, r2
    except Exception as e:
        return 'No sequence', 'No Sequence', 'No sequence', 'No Sequence'


def process_per_segment(whos, test, result):
    answer =[[whos, whos, whos, whos]]
    answer.append(['RMSE', 'MAE', 'COR', 'R2'])

    for i in range(len(test)):
        rmse, mae, cor, r2 = evaluation(test[i], result[i])
        answer.append([rmse, mae, cor, r2])
    rmse, mae, cor, r2 = evaluation(test, result)
    answer.append([rmse, mae, cor, r2])

    return answer
