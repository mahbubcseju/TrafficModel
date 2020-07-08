import numpy as np

from statsmodels.tsa.arima_model import ARIMA

from utils import preprocess_data_config, evaluation


def model_output(data, prelen, p, d, q):
    try:
        model = ARIMA(data, order=[p, d, q])
        trained_model = model.fit(disp=-1)
        output = trained_model.forecast(prelen)[0]
        print(data, "output", output)
        return output, 1
    except Exception as e:
        print(e, data)
        return [0] * prelen, 0


def arima_sampling(train, test, rate=0.5, seq_len=12, sampling_rate=2, pre_len=3, repeat=False, is_continuous=True, p=2, d=1, q=3):
    train_x, train_y, header = preprocess_data_config(train,  seq_len, sampling_rate=sampling_rate, pre_len=pre_len)
    test_x, test_y, header = preprocess_data_config(test, seq_len, sampling_rate=sampling_rate, pre_len=pre_len)

    num_nodes = len(header)

    total_test_Y, total_predict_Y = [], []
    count_invalid, total = 0, 0
    for i in range(num_nodes):
        if i !=26:
            continue
        a_X, a_Y = train_x[:, :, i], train_y[:, :, i]
        t_X, t_Y = test_x[:, :, i], test_y[:, :, i]

        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, pre_len])

        result_y = []
        test1_y = []
        for i in range(len(t_X)):
            a = np.array(t_X[i])
            if np.sum(a) == 0:
                continue

            if repeat:
                output, is_valid = model_output(a, 1, p, d, q)
                temp_result = [output[0] for i in range(pre_len)]
            else:
                temp_result, is_valid = model_output(a, pre_len, p, d, q)

            total += 1
            if is_valid:
                result_y.append(temp_result)
                test1_y.append(t_Y[i].tolist())
            else:
                count_invalid += 1

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

    # test1 = np.reshape(np.array(total_test_Y), [num_nodes, -1])
    # result1 = np.reshape(np.array(total_predict_Y), [num_nodes, -1])

    return header, total_test_Y, total_predict_Y, count_invalid, total
