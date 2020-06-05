import numpy.linalg as la
import math

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr


def preprocess_data(data, rate, seq_len=12, pre_len=3):
    np_data = np.array(data)
    time_len = np_data.shape[0]

    data_x, data_y = [], []
    for i in range(0, time_len - seq_len - pre_len):
        data_x.append(np_data[i:i+seq_len])
        data_y.append(np_data[i+seq_len:i+seq_len+pre_len])

    train_size = int(time_len * rate)
    train_x, train_y = data_x[:train_size], data_y[:train_size]
    test_x, test_y = data_x[train_size:], data_y[train_size:]
    return train_x, train_y, test_x, test_y


###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b)/la.norm(a)
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a - b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var
