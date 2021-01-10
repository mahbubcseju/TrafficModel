import os
import csv
import time
import pandas as pd
import numpy as np


# Configurations
prediction_length = 9
start_time = '08:00'
end_time = '09:00'

##############################


current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'csvs')
models = ['HA', 'SVR', 'SVR_GRAPH', 'ARIMA']


for model in models:
    file_path = os.path.join(data_path, f'{model}_{5 * prediction_length}.csv')
    data = pd.read_csv(file_path)
    data = data.set_index('time')
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
    data = data['2019-11-24 12:05:00':]
    data['time_only'] = data.index.map(lambda x: str(x).split()[1])
    filtered_data = data.loc[(start_time <= data['time_only']) & (data['time_only'] <= end_time)]
    print(filtered_data.columns)
    break