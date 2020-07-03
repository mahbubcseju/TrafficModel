import numpy as np

from utils import (
    preprocess_data_config,
    evaluation,
)


def ha_sampling(train, test, seq_len=12, sampling_rate=2, pre_len=3, repeat=False, is_continuous=True):
    train_x, train_y, header = preprocess_data_config(train,  seq_len, sampling_rate=sampling_rate, pre_len=pre_len)
    test_x, test_y, header = preprocess_data_config(test, seq_len, sampling_rate=sampling_rate, pre_len=pre_len)

    num_nodes = len(header)
    total_test_Y, total_predict_Y = [], []
    for i in range(num_nodes):
        a_X, a_Y = train_x[:, :, i], train_y[:, :, i]
        t_X, t_Y = test_x[:, :, i], test_y[:, :, i]

        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, pre_len])

        result_y = []
        test1_y = []
        for i in range(len(t_X)):
            a = np.array(t_X[i], dtype='float')
            if np.sum(a) == 0:
                continue

            test1_y.append(t_Y[i].tolist())

            if repeat:
                output = np.mean(a)
                temp_result = [output for i in range(pre_len)]
                result_y.append(temp_result)
            else:
                temp_result = []
                for nxt in range(pre_len):
                    prediction = np.mean(a)
                    temp_result.append(prediction)
                    a = np.append(a, prediction)
                    a = a[1:]
                result_y.append(temp_result)

        t_Y = test1_y

        if not is_continuous:
            temp_test_y = []
            temp_result_y = []
            for i in range(len(t_Y)):
                temp_test_y.append(t_Y[i][pre_len - 1])
                temp_result_y.append(result_y[i][pre_len - 1])
            t_Y = temp_test_y
            result_y = temp_result_y

        total_test_Y.append(t_Y)
        total_predict_Y.append(result_y)

    # test1 = np.reshape(np.array(total_test_Y), (num_nodes, -1)).tolist()
    # result1 = np.reshape(np.array(total_predict_Y), (num_nodes, -1)).tolist()

    return header, total_test_Y, total_predict_Y


def ha_sampling1(train, test, seq_len=12, sampling_rate=2, pre_len=3, repeat=False, is_continuous=True):

    train_x, train_y, header = preprocess_data_config(train,  seq_len, sampling_rate=sampling_rate, pre_len=pre_len)
    test_x, test_y, header = preprocess_data_config(test, seq_len, sampling_rate=sampling_rate, pre_len=pre_len)

    result_y = []
    for i in range(len(test_x)):
        a = np.array(test_x[i], dtype='float')

        if repeat:
            a_mean = np.mean(a, axis=0)
            temp_result = []
            for i in range(pre_len):
                temp_result.append(a_mean)
            result_y.append(np.array(temp_result))
        else:
            temp_result = []
            for nxt in range(pre_len):
                a_mean = np.mean(a, axis=0)
                temp_result.append(a_mean)
                a = np.append(a, [a_mean], axis=0)
                a = a[1:]

            result_y.append(np.array(temp_result))

    if not is_continuous:
        temp_test_y = []
        temp_result_y = []
        for i in range(len(test_y)):
            temp_test_y.append(test_y[i][pre_len-1])
            temp_result_y.append(result_y[i][pre_len-1])
        test_y = temp_test_y
        result_y = temp_result_y

    test_y1 = np.array(test_y, dtype='float')
    result_y1 = np.array(result_y, dtype='float')
    num_nodes = test_y1.shape[-1]
    result1 = np.reshape(result_y1, [-1, num_nodes]).T
    test1 = np.reshape(test_y1, [-1, num_nodes]).T

    return header, test1, result1
