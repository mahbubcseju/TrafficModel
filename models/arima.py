import numpy as np

from statsmodels.tsa.arima_model import ARIMA

from utils import preprocess_data, evaluation


def model_output(data):
    try:
        model = ARIMA(data, order=[1, 0, 0])
        trained_model = model.fit(disp=-1)
        output = trained_model.forecast()
        return output[0][0]
    except Exception:
        return 0


def arima(data, rate=0.5, seq_len=12, pre_len=3, repeat=False, is_continuous=True):
    data = np.mat(data)
    num_nodes = data.shape[1]

    total_test_Y, total_predict_Y = [], []
    for i in range(num_nodes):
        node_data = data[1:, i]
        a_X, a_Y, t_X, t_Y = preprocess_data(node_data, rate=rate, seq_len=seq_len, pre_len=pre_len)

        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, pre_len])

        result_y = []
        for i in range(len(t_X)):
            a = np.array(t_X[i])
            if repeat:
                output = model_output(a)
                temp_result = [output for i in range(pre_len)]
                result_y.append(temp_result)
            else:
                temp_result = []
                for nxt in range(pre_len):
                    prediction = model_output(a)
                    temp_result.append(prediction)
                    a = np.append(a, prediction)
                    a = a[1:]
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
    rmse, mae, accuracy, r2, var = evaluation(test1, result1)
    print('ARIMA_rmse:%r' % rmse,
          'ARIMA_mae:%r' % mae,
          'ARIMA_acc:%r' % accuracy,
          'ARIMA_r2:%r' % r2,
          'ARIMA_var:%r' % var)
