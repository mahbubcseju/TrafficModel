import os

import pandas as pd

from models import (
    ha,
    svr,
    arima,
    ha_sampling,
    svr_sampling,
    arima_sampling,
)

current_directory = os.getcwd()

data_set = '13-11-19_mirpurin_out.csv'

data_path = os.path.join(current_directory, 'csvs', data_set)
data = pd.read_csv(data_path).T

ha(data, repeat=False, is_continuous=True)
# ha_sampling(data, repeat=False, is_continuous=True, sampling_rate=3)

# svr(data, repeat=False, is_continuous=True)

# arima(data, repeat=False, is_continuous=True)
arima_sampling(data, repeat=False, is_continuous=True, sampling_rate=3)