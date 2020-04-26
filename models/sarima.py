import numpy as np

import pyramid
import pmdarima as pm

from utils.preprocess_data import preprocess_data, evaluation


def model_output(data, prelen):
    try:
        sarima_model = pm.auto_arima(data, start_p=1, start_q=1, max_p=8, max_q=8, start_P=0, start_Q=0, max_P=8, max_Q=8,
                                 m=12, seasonal=True, trace=True, d=1, D=1, error_action='warn', suppress_warnings=True,
                                 random_state=20, n_fits=30)
        predictions = sarima_model.predict(n_periods=prelen)
        return predictions
    except Exception:
        return [0] * prelen

def sarima(data, rate=0.5, seq_len=12, pre_len=3, repeat=False, is_continuous=True):
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
                output = model_output(a, 1)
                temp_result = [output[0] for i in range(pre_len)]
                result_y.append(temp_result)
            else:
                temp_result = model_output(a, pre_len)
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
