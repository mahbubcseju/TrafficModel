import os
import csv

import pandas as pd
import numpy as np

from models import (
    arima_sampling,
    ha,
    svr,
    arima,
    ha_sampling,
    svr_sampling,
    svr_sampling_graph,
)

from utils import process_per_segment


def run_models(base_directory, train_nt, test_nt, sampling_rate=2, seq_len=60, pre_len=10, repeat=False, is_continuous=False):
    # ha(data, pre_len=1, repeat=False, is_continuous=True)
    result = [['', ''],['Node Number', 'Node name']]
    ans = []

    # train, test = np.array(train).T, np.array(test).T
    train = [np.array(data).T for data in train_nt]
    test = [ np.array(data).T for data in test_nt]

    # ha_header, ha_test, ha_result = ha_sampling(train, test, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate)
    # for i in range(len(ha_header)):
    #     result.append([i, ha_header[i]])
    # result.append(['Average', ''])
    # result = np.array(result)
    #
    # ha_temp_result = process_per_segment('HA', ha_test, ha_result)
    # result = np.concatenate([result, ha_temp_result], axis=1)
    # ha_avg = [sampling_rate, seq_len, pre_len, 'HA']
    # for i in ha_temp_result[-1]:
    #     ha_avg.append(i)
    # ans.append(ha_avg)

    print('HA Complete')

    # svr_header, svr_test, svr_result = svr_sampling(train, test, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate)
    # svr_temp_result = process_per_segment('SVR', svr_test, svr_result)
    # result = np.concatenate([result, svr_temp_result], axis=1)
    #
    # svr_avg = [sampling_rate, seq_len, pre_len, 'SVR']
    # for i in svr_temp_result[-1]:
    #     svr_avg.append(i)
    # ans.append(svr_avg)

    print('SVR Complete')

    # # arima(data, pre_len=3, repeat=False, is_continuous=True)
    # # arima_sampling(data, seq_len=60, pre_len=5, repeat=False, is_continuous=False, sampling_rate=2)
    # #
    # # sarima(data, rate=0.5, seq_len=12, pre_len=3, repeat=False, is_continuous=True)
    #
    #
    adjacent_path = os.path.join(base_directory, 'csvs', 'adjacencyMatrixUpdated.csv')
    adjacency_matrix = pd.read_csv(adjacent_path, index_col=0)

    # intersection_name_from_adj = adjacency_matrix.columns.values
    # intersection_name_from_data = list(svr_header)
    # print(intersection_name_from_adj, intersection_name_from_data)
    # print(all(intersection_name_from_adj == intersection_name_from_data))
    #
    # svr_graph_header, svr_graph_test, svr_graph_result = svr_sampling_graph(train, test, adjacency_matrix, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate)
    # svr_graph_temp_result = process_per_segment('SVR GRAPH', svr_graph_test, svr_graph_result)
    # result = np.concatenate([result, svr_graph_temp_result], axis=1)
    # _avg = [sampling_rate, seq_len, pre_len, 'SVR_GRAPH']
    # for i in svr_graph_temp_result[-1]:
    #     _avg.append(i)
    # ans.append(_avg)
    # print('SVR GRAPH Complete')

    arima_header, arima_test, arima_result, count_invalid, total = arima_sampling(train, test, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate, p=1, d=1, q=1)
    arima_temp_result = process_per_segment('ARIMA', arima_test, arima_result)
    print(arima_temp_result)
    result = np.concatenate([result, arima_temp_result], axis=1)
    _avg = [sampling_rate, seq_len, pre_len, 'ARIMA']
    for i in arima_temp_result[-1]:
        _avg.append(i)
    _avg.append(str(count_invalid) + "/" + str(total))
    ans.append(_avg)
    print('Arima complete')

    final_result = [
        ['Date', '29-05-2019'],
        ['parameters', 'Sampling Rate', 'Sequence length', 'Prediction Length', 'Repeat', 'Is Continuous'],
        ['values', sampling_rate, seq_len, pre_len, repeat, is_continuous],
    ]

    for row  in  result:
        final_result.append(row)

    file_name = 'Sam-{}-seq-{}-pre-{}-repeat-{}-is-continous-{}.csv'.format(
        sampling_rate,
        seq_len,
        pre_len,
        repeat,
        is_continuous,
    )
    image_data_csv_file = os.path.join(base_directory, 'csvs', file_name )

    with open(image_data_csv_file, 'w') as writer:
        wr = csv.writer(writer)
        wr.writerows(final_result)

    print('FINAL COMPLETE')

    return ans