import numpy.linalg as la
import math

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error


def preprocess_data_sampling(data, rate, seq_len=12, sampling_rate=2, pre_len=3):
    np_data = np.array(data)
    time_len = np_data.shape[0]

    data_x, data_y = [], []
    per_seq_covered_length = (seq_len + pre_len - 1) * (sampling_rate - 1) + seq_len + pre_len
    for i in range(0, time_len - per_seq_covered_length):
        data_x.append([np_data[j] for j in range(i, i + seq_len * sampling_rate, sampling_rate)])
        data_y.append(
            [
                np_data[j] for j in range(
                    i + seq_len * sampling_rate, i + (seq_len + pre_len) * sampling_rate, sampling_rate
                )
            ]
        )

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


def preprocess_data_sampling_graph(data, rate, seq_len=12, sampling_rate=2, pre_len=3):
    np_data = np.array(data)
    time_len = np_data.shape[0]

    data_x, data_y = [], []
    per_seq_covered_length = (seq_len + pre_len - 1) * (sampling_rate - 1) + seq_len + pre_len
    for i in range(0, time_len - per_seq_covered_length):
        data_x.append([np_data[j] for j in range(i, i + seq_len * sampling_rate, sampling_rate)])
        data_y.append(
            [
                np_data[j][-1] for j in range(
                    i + seq_len * sampling_rate, i + (seq_len + pre_len) * sampling_rate, sampling_rate
                )
            ]
        )
    # print(len(data_x), len(data_y))
    # print(data_x, data_y)
    train_size = int(time_len * rate)
    train_x, train_y = data_x[:train_size], data_y[:train_size]
    test_x, test_y = data_x[train_size:], data_y[train_size:]
    return train_x, train_y, test_x, test_y
