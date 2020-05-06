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
    sarima,
)

current_directory = os.getcwd()

data_set = '13-11-19_mirpurin_out.csv'

data_path = os.path.join(current_directory, 'csvs', data_set)
data = pd.read_csv(data_path, index_col=0).T


# start_index = '13_11_2019 06_00_01_AM'
# end_index = '13_11_2019 08_01_31_AM'
# data = data.loc[start_index: end_index]

# ha(data, pre_len=1, repeat=False, is_continuous=True)
# # ha_sampling(data, repeat=False, is_continuous=True, sampling_rate=3)
#
svr(data, pre_len=1, repeat=False, is_continuous=True)

# arima(data, pre_len=3, repeat=False, is_continuous=True)
# arima_sampling(data, repeat=False, is_continuous=True, sampling_rate=3)
#
# sarima(data, rate=0.5, seq_len=12, pre_len=3, repeat=False, is_continuous=True)


adjacent_path = os.path.join(current_directory, 'csvs', 'adjacencyMatrix.csv')
adjacency_matrix = pd.read_csv(adjacent_path, index_col=0)

intersection_name_from_adj = adjacency_matrix.columns.values
intersection_name_from_data = list(data.columns.values)

print(all(intersection_name_from_adj == intersection_name_from_data))

svr_sampling_graph(data, adjacency_matrix)
