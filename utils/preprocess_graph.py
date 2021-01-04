import numpy as np


def preprocess_data(df, sequence_length, prediction_length):
    values = df.to_numpy()

    length = len(values)
    data_x, data_y = [], []
    for i in range(length - prediction_length - sequence_length):
        data_x.append(values[i: i + sequence_length])
        data_y.append(values[i + sequence_length: i + sequence_length + prediction_length])

    return np.array(data_x), np.array(data_y)
