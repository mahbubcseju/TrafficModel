import os

import pandas as pd

from models import (
    arima_sampling,
    ha,
    svr,
    arima,
    ha_sampling,
    svr_sampling,
    svr_sampling_graph,
)

current_directory = os.getcwd()

data_set = 'novemberin_out.csv'

data_path = os.path.join(current_directory, 'csvs', data_set)
data = pd.read_csv(data_path, index_col=0).T



# start_index = '13_11_2019 06_00_01_AM'
# end_index = '13_11_2019 08_01_31_AM'
# data = data.loc[start_index: end_index]

sampling_rate = 2
seq_len = 120
pre_len = 5
repeat= True
is_continuous = False


# ha(data, pre_len=1, repeat=False, is_continuous=True)
#ha_sampling(data, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate)
#
#svr_sampling(data, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate)

# arima(data, pre_len=3, repeat=False, is_continuous=True)
#arima_sampling(data, seq_len=60, pre_len=5, repeat=False, is_continuous=False, sampling_rate=2)
#
# sarima(data, rate=0.5, seq_len=12, pre_len=3, repeat=False, is_continuous=True)


adjacent_path = os.path.join(current_directory, 'csvs', 'adjacencyMatrix.csv')
adjacency_matrix = pd.read_csv(adjacent_path, index_col=0)

intersection_name_from_adj = adjacency_matrix.columns.values
intersection_name_from_data = list(data.columns.values)

print(all(intersection_name_from_adj == intersection_name_from_data))

#svr_sampling_graph(data, adjacency_matrix, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate)

arima_sampling(data, seq_len=seq_len, pre_len=pre_len, repeat=repeat, is_continuous=is_continuous, sampling_rate=sampling_rate)
