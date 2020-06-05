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

current_directory = os.getcwd()

data_set = 'november-2019in_out.csv'

data_path = os.path.join(current_directory, 'csvs', data_set)
data = pd.read_csv(data_path, index_col=0).T



# start_index = '13_11_2019 06_00_01_AM'
# end_index = '13_11_2019 08_01_31_AM'
# data = data.loc[start_index: end_index]

sampling_rate = 2
seq_len = 60
pre_len = 10
repeat = False
is_continuous = False


# ha(data, pre_len=1, repeat=False, is_continuous=True)
result = [['', ''],['Node Number', 'Node name']]

ha_header, ha_test, ha_result = ha_sampling(data, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate)

for i in range(len(ha_header)):
    result.append([i, ha_header[i]])
result.append(['Average', ''])
result = np.array(result)

ha_temp_result = process_per_segment('HA', ha_test, ha_result)
result = np.concatenate([result, ha_temp_result], axis=1)

print('HA Complete')

svr_header, svr_test, svr_result = svr_sampling(data, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate)
svr_temp_result = process_per_segment('SVR', svr_test, svr_result)
result = np.concatenate([result, svr_temp_result], axis=1)


print('SVR Complete')
# arima(data, pre_len=3, repeat=False, is_continuous=True)
# arima_sampling(data, seq_len=60, pre_len=5, repeat=False, is_continuous=False, sampling_rate=2)
#
# sarima(data, rate=0.5, seq_len=12, pre_len=3, repeat=False, is_continuous=True)


adjacent_path = os.path.join(current_directory, 'csvs', 'adjacencyMatrix.csv')
adjacency_matrix = pd.read_csv(adjacent_path, index_col=0)

intersection_name_from_adj = adjacency_matrix.columns.values
intersection_name_from_data = list(data.columns.values)

print(all(intersection_name_from_adj == intersection_name_from_data))

svr_graph_header, svr_graph_test, svr_graph_result = svr_sampling_graph(data, adjacency_matrix, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate)
svr_graph_temp_result = process_per_segment('SVR GRAPH', svr_graph_test, svr_graph_result)
result = np.concatenate([result, svr_graph_temp_result], axis=1)
print('SVR GRAPH Complete')
arima_header, arima_test, arima_result = arima_sampling(data, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate, p=1, d=1, q=1)
arima_temp_result = process_per_segment('ARIMA', arima_test, arima_result)
result = np.concatenate([result, arima_temp_result], axis=1)


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
image_data_csv_file = os.path.join(current_directory, 'csvs', file_name )

with open(image_data_csv_file, 'w') as writer:
    wr = csv.writer(writer)
    wr.writerows(final_result)

print('FINAL COMPLETE')