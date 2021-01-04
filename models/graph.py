import numpy as np
from sklearn.svm import SVR

from statsmodels.tsa.arima_model import ARIMA

from utils import preprocess_data


def ha_graph(train, test, sequence_length=15, prediction_length=3):
    header = train.columns
    train_x, train_y = preprocess_data(train, sequence_length=sequence_length, prediction_length=prediction_length)
    test_x, test_y = preprocess_data(test, sequence_length=sequence_length, prediction_length=prediction_length)

    result_y = []
    for i in range(len(test_x)):
        a = np.array(test_x[i], dtype='float')
        temp_result = []
        for nxt in range(prediction_length):
            a_mean = np.mean(a, axis=0)
            temp_result.append(a_mean)
            a = np.append(a, [a_mean], axis=0)
            a = a[1:]

        result_y.append(np.array(temp_result))

    temp_test_y = []
    temp_result_y = []
    for i in range(len(test_y)):
        temp_test_y.append(test_y[i][prediction_length - 1])
        temp_result_y.append(result_y[i][prediction_length - 1])
    test_y = temp_test_y
    result_y = temp_result_y

    test_y1 = np.array(test_y, dtype='float')
    result_y1 = np.array(result_y, dtype='float')
    num_nodes = test_y1.shape[-1]
    result1 = np.reshape(result_y1, [-1, num_nodes]).T
    test1 = np.reshape(test_y1, [-1, num_nodes]).T

    return header, test1, result1


def svr_graph(train, test, sequence_length, prediction_length):
    header = train.columns
    train_x, train_y = preprocess_data(train, sequence_length=sequence_length, prediction_length=prediction_length)
    test_x, test_y = preprocess_data(test, sequence_length=sequence_length, prediction_length=prediction_length)

    num_nodes = len(header)
    total_test_Y, total_predict_Y = [], []
    for i in range(num_nodes):
        a_X, a_Y = train_x[:, :, i], train_y[:, :, i]
        t_X, t_Y = test_x[:, :, i], test_y[:, :, i]
        a_X = np.array(a_X)
        a_X = np.reshape(a_X, [-1, sequence_length])
        a_Y = np.array(a_Y)
        a_Y = a_Y[:, 0]
        a_Y = a_Y.flatten()

        final_x, final_y = [], []
        for i in range(len(a_X)):
            if np.sum(a_X[i]) == 0:
                continue
            final_x.append(a_X[i])
            final_y.append(a_Y[i])
        a_X = np.array(final_x)
        a_Y = np.array(final_y)
        try:

            result_y = []
            test1_y = []

            model = SVR(kernel='rbf')
            model = model.fit(a_X, a_Y)
        except Exception as e:
            print(e)

        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, sequence_length])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, prediction_length])

        for i in range(len(t_X)):
            a = np.array(t_X[i])
            test1_y.append(t_Y[i].tolist())

            if np.sum(a) == 0:
                result_y.append([0] * prediction_length)
                continue
            try:
                temp_result = []
                for nxt in range(prediction_length):
                    prediction = model.predict([a])
                    temp_result.append(prediction[0])
                    a = np.append(a, prediction[0])
                    a = a[1:]
                result_y.append(temp_result)
            except Exception as e:
                result_y.append([0] * prediction_length)


        t_Y = test1_y
        temp_test_y = []
        temp_result_y = []
        for i in range(len(t_Y)):
            temp_test_y.append(t_Y[i][prediction_length - 1])
            temp_result_y.append(result_y[i][prediction_length - 1])
        t_Y = temp_test_y
        result_y = temp_result_y

        total_test_Y.append(t_Y)
        total_predict_Y.append(result_y)

    # test1 = np.reshape(np.array(total_test_Y), [num_nodes, -1])
    # result1 = np.reshape(np.array(total_predict_Y), [num_nodes, -1])

    return header, total_test_Y, total_predict_Y


def svr_graph_graph(train, test, sequence_length=None,prediction_length= None):
    header = train.columns
    train_x, train_y = preprocess_data(train, sequence_length=sequence_length, prediction_length=prediction_length)
    test_x, test_y = preprocess_data(test, sequence_length=sequence_length, prediction_length=prediction_length)

    def get_adjacent(i, headers):
        end_node = headers[i].split('-')[-1]
        return [j for j in range(len(headers)) if headers[j].split('-')[-1] == end_node]

    num_nodes = len(header)
    total_test_Y, total_predict_Y = [], []
    for i in range(num_nodes):
        adjacent = get_adjacent(i, header)
        total_adjacent = len(adjacent)
        a_X, a_Y = train_x[:, :, adjacent], train_y[:, :, [i]]
        t_X, t_Y = test_x[:, :, adjacent], test_y[:, :, [i]]

        a_X = np.array(a_X)
        a_X = np.reshape(a_X, [-1, sequence_length * total_adjacent])
        a_Y = np.array(a_Y)
        a_Y = a_Y[:, 0]
        a_Y = a_Y.flatten()

        final_x, final_y = [], []
        for i in range(len(a_X)):
            if np.sum(a_X[i]) == 0:
                continue
            final_x.append(a_X[i])
            final_y.append(a_Y[i])
        a_X = np.array(final_x)
        a_Y = np.array(final_y)
        try:

            result_y = []
            test1_y = []

            model = SVR(kernel='rbf')
            model = model.fit(a_X, a_Y)

            t_X = np.array(t_X)
            t_X = np.reshape(t_X, [-1, sequence_length * total_adjacent])
            t_Y = np.array(t_Y)
            t_Y = np.reshape(t_Y, [-1, prediction_length])
        except Exception as e:
            print(e)

        for i in range(len(t_X)):
            a = np.array(t_X[i])
            test1_y.append(t_Y[i].tolist())
            if np.sum(a) == 0:
                result_y.append([0] * prediction_length)
                continue
            try:
                temp_result = []
                for nxt in range(prediction_length):
                    prediction = model.predict([a])
                    temp_result.append(prediction[0])
                    a = np.append(a, prediction[0])
                    a = a[1:]
                result_y.append(temp_result)
            except Exception as e:
                result_y.append([0] * prediction_length)


        t_Y = test1_y
        temp_test_y = []
        temp_result_y = []
        for i in range(len(t_Y)):
            temp_test_y.append(t_Y[i][prediction_length - 1])
            temp_result_y.append(result_y[i][prediction_length - 1])
        t_Y = temp_test_y
        result_y = temp_result_y

        total_test_Y.append(t_Y)
        total_predict_Y.append(result_y)

        # test1 = np.reshape(np.array(total_test_Y), [num_nodes, -1])
        # result1 = np.reshape(np.array(total_predict_Y), [num_nodes, -1])

    return header, total_test_Y, total_predict_Y


def model_output(data, prelen, p, d, q):
    try:
        model = ARIMA(data, order=[p, d, q])
        trained_model = model.fit(disp=-1, maxiter=200)
        output = trained_model.forecast(prelen)[0]
        if max(abs(output)) >= 1500:
            return output, 0
        return output, 1
    except Exception as e:
        return [0] * prelen, 0


def arima_graph(train, test, sequence_length, prediction_length, p=1, d=0, q=0):
    header = train.columns
    train_x, train_y = preprocess_data(train, sequence_length=sequence_length, prediction_length=prediction_length)
    test_x, test_y = preprocess_data(test, sequence_length=sequence_length, prediction_length=prediction_length)

    num_nodes = len(header)
    total_test_Y, total_predict_Y = [], []
    count_invalid, total = 0, 0
    for i in range(num_nodes):
        a_X, a_Y = train_x[:, :, i], train_y[:, :, i]
        t_X, t_Y = test_x[:, :, i], test_y[:, :, i]

        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, sequence_length])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, prediction_length])

        result_y = []
        test1_y = []
        for i in range(len(t_X)):
            a = np.array(t_X[i])
            test1_y.append(t_Y[i].tolist())
            if np.sum(a) == 0:
                result_y.append([0] * prediction_length)
                continue

            temp_result, is_valid = model_output(a, prediction_length, p, d, q)

            total += 1
            if is_valid:
                result_y.append(temp_result)
            else:
                result_y.append(t_Y[i].toList())
                count_invalid += 1

        t_Y = test1_y

        temp_test_y = []
        temp_result_y = []
        for i in range(len(t_Y)):
            temp_test_y.append(t_Y[i][prediction_length - 1])
            temp_result_y.append(result_y[i][prediction_length - 1])
        t_Y = temp_test_y
        result_y = temp_result_y

        total_test_Y.append(t_Y)
        total_predict_Y.append(result_y)

    # test1 = np.reshape(np.array(total_test_Y), [num_nodes, -1])
    # result1 = np.reshape(np.array(total_predict_Y), [num_nodes, -1])

    return header, total_test_Y, total_predict_Y, count_invalid, total
