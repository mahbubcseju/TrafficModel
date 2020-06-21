import numpy as np

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

from utils import preprocess_data_config, evaluation


def svr_sampling(train, test, rate=0.5, seq_len=12, sampling_rate=2, pre_len=3, repeat=False, is_continuous=True):
    train_x, train_y, header = preprocess_data_config(train,  seq_len, sampling_rate=sampling_rate, pre_len=pre_len)
    test_x, test_y, header = preprocess_data_config(test, seq_len, sampling_rate=sampling_rate, pre_len=pre_len)

    num_nodes = len(header)
    total_test_Y, total_predict_Y = [], []
    for i in range(num_nodes):
        a_X, a_Y = train_x[:, :, i], train_y[:, :, i]
        t_X, t_Y = test_x[:, :, i], test_y[:, :, i]
        a_X = np.array(a_X)
        a_X = np.reshape(a_X, [-1, seq_len])
        a_Y = np.array(a_Y)
        a_Y = a_Y[:, 0]
        a_Y = a_Y.flatten()

        model = SVR(kernel='rbf')
        model = model.fit(a_X, a_Y)

        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, pre_len])

        result_y = []
        for i in range(len(t_X)):
            a = np.array(t_X[i])
            if repeat:
                prediction = model.predict([a])
                temp_result = [prediction[0] for i in range(pre_len)]
                result_y.append(temp_result)
            else:
                temp_result = []
                for nxt in range(pre_len):
                    prediction = model.predict([a])
                    temp_result.append(prediction[0])
                    a = np.append(a, prediction[0])
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

    return header, test1, result1
