import os

import pandas as pd

from models import HA, SVR

current_directory = os.getcwd()

data_set = '13-11-19_mirpurin_out.csv'

data_path = os.path.join(current_directory, 'csvs', data_set)
data = pd.read_csv(data_path).T

# HA(data, repeat=False, is_continuous=True)

SVR(data, repeat=False, is_continuous=True)



