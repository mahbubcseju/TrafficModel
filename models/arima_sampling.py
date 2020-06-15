import numpy as np

from statsmodels.tsa.arima_model import ARIMA

from utils import preprocess_data_config, evaluation


def model_output(data, prelen, p, d, q):
    try:
        model = ARIMA(data, order=[p, d, q])
        trained_model = model.fit(disp=-1)
        output = trained_model.forecast(prelen)[0]
        return output
    except Exception:
        return [0] * prelen


def arima_sampling(train, test, rate=0.5, seq_len=12, sampling_rate=2, pre_len=3, repeat=False, is_continuous=True, p=2, d=1, q=3):
    header = train[0][1:]
    train, test = np.array(train[1:, 1:], dtype='float'), np.array(test[1:, 1:], dtype='float')
    num_nodes = train.shape[1]
    num_nodes = train.shape[1]

    total_test_Y, total_predict_Y = [], []
    for i in range(num_nodes):
        node_train = train[:, i]
        node_test = test[:, 1]
        a_X, a_Y = preprocess_data_config(node_train, seq_len, sampling_rate=sampling_rate, pre_len=pre_len)
        t_X, t_Y = preprocess_data_config(node_test, seq_len, sampling_rate=sampling_rate, pre_len=pre_len)

        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, pre_len])

        result_y = []
        for i in range(len(t_X)):
            a = np.array(t_X[i])
            if repeat:
                output = model_output(a, 1, p, d, q)
                temp_result = [output[0] for i in range(pre_len)]
                result_y.append(temp_result)
            else:
                temp_result = model_output(a, pre_len, p, d, q)
                result_y.append(np.array(temp_result))

        if not is_continuous:
            temp_test_y = []
            temp_result_y = []
            for i in range(len(t_Y)):
                temp_test_y.append(t_Y[i][pre_len - 1])
                temp_result_y.append(result_y[i][pre_len - 1])
            t_Y = temp_test_y
            result_y = temp_result_y

        total_test_Y.append(np.array(t_Y))
        total_predict_Y.append(np.array(result_y))

    test1 = np.reshape(np.array(total_test_Y), [num_nodes, -1])
    result1 = np.reshape(np.array(total_predict_Y), [num_nodes, -1])

    return header, test1, result1
