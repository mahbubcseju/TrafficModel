import numpy as np

from utils import preprocess_data_sampling, evaluation


def ha_sampling(data, rate=0.5, seq_len=12, sampling_rate=2, pre_len=3, repeat=False, is_continuous=True):
    train_x, train_y, test_x, test_y = preprocess_data_sampling(data, rate, seq_len, sampling_rate=sampling_rate, pre_len=pre_len)

    result_y = []
    for i in range(len(test_x)):
        a = np.array(test_x[i])

        if repeat:
            a_mean = np.mean(a, axis=0)
            a1 = a_mean.repeat(pre_len, axis=0)
            result_y.append(a1)
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

    test_y1 = np.array(test_y)
    result_y1 = np.array(result_y)
    num_nodes = test_y1.shape[-1]
    result1 = np.reshape(result_y1, [-1, num_nodes])
    test1 = np.reshape(test_y1, [-1, num_nodes])

    rmse, mae, accuracy,r2,var = evaluation(test1, result1)
    print('HA_rmse:%r'%rmse,
          'HA_mae:%r'%mae,
          'HA_acc:%r'%accuracy,
          'HA_r2:%r'%r2,
          'HA_var:%r'%var)
