import os

import pandas as pd

from models import ha, svr, arima

current_directory = os.getcwd()

data_set = '13-11-19_mirpurin_out.csv'

data_path = os.path.join(current_directory, 'csvs', data_set)
data = pd.read_csv(data_path).T

# ha(data, repeat=False, is_continuous=True)

# svr(data, repeat=False, is_continuous=True)

arima(data, repeat=False, is_continuous=True)
