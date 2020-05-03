import os

import pandas as pd

from models import (
    ha,
    svr,
    arima,
    ha_sampling,
    svr_sampling,
    sarima,
    arima_sampling,
)

current_directory = os.getcwd()

data_set = '13-11-19_mirpurin_out.csv'

data_path = os.path.join(current_directory, 'csvs', data_set)
data = pd.read_csv(data_path).T

start_index = '13_11_2019 06_00_01_AM'
end_index = '13_11_2019 08_01_31_AM'
data = data.loc[start_index: end_index]

ha(data, pre_len=1, repeat=False, is_continuous=True)
# # ha_sampling(data, repeat=False, is_continuous=True, sampling_rate=3)
#
# svr(data, pre_len=1, repeat=False, is_continuous=True)

# arima(data, pre_len=3, repeat=False, is_continuous=True)
# arima_sampling(data, repeat=False, is_continuous=True, sampling_rate=3)
#
# sarima(data, rate=0.5, seq_len=12, pre_len=3, repeat=False, is_continuous=True)