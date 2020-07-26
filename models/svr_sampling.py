import numpy as np

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
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


        a_Y = a_Y[:, pre_len - 1]
        a_Y = a_Y.flatten()

        final_x, final_y = [], []
        for i in range(len(a_X)):
            if np.sum(a_X[i]) == 0:
                continue
            final_x.append(a_X[i])
            final_y.append(a_Y[i])
        a_X = np.array(final_x)
        a_Y = np.array(final_y)

        model = SVR(kernel='rbf', max_iter=1000)
        model = model.fit(a_X, a_Y)

        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, pre_len])
        t_Y = t_Y[:, pre_len - 1]
        t_Y = t_Y.flatten()

        result_y = []
        test1_y = []
        for i in range(len(t_X)):
            a = np.array(t_X[i])

            if np.sum(a) == 0:
                continue

            test1_y.append(t_Y[i])
            temp_result = model.predict([a])
            result_y.append(temp_result[0])

        t_Y = test1_y
        total_test_Y.append(t_Y)
        total_predict_Y.append(result_y)

    return header, total_test_Y, total_predict_Y
