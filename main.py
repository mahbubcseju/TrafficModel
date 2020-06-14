from config import config

for key in config['train']:
    value = config['train'][key]
    start_date = value['start_date'] if 'start_date' in value else '01_11_2019'
    end_date = value['end_date'] if 'end_date' in value else '01_11_2019'
    start_time = value['start_time'] if 'start_time' in value else '01_11_2019'
    end_time = value['end_time'] if 'end_time' in value else '01_11_2019'

    start, end = int(start_time.split('_')[0])



for key in config['test']:
    value = config['test'][key]
    start_date = value['start_date'] if 'start_date' in value else '01_11_2019'
    end_date = value['end_date'] if 'end_date' in value else '01_11_2019'
    start_time = value['start_time'] if 'start_time' in value else '01_11_2019'
    end_time = value['end_time'] if 'end_time' in value else '01_11_2019'

    print(start_date, end_date, start_time, end_time)


import os
import csv

import pandas as pd
import numpy as np

current_directory = os.getcwd()

data_set = 'november0.csv'

data_path = os.path.join(current_directory, 'csvs', data_set)
data = pd.read_csv(data_path, index_col=0)

start_index = '01_11_2019 06_00_01'
end_index = '01_11_2019 06_01_31'
data = data.loc[:, start_index: end_index]
print(data.columns.values)
print(data.index)
print(data)
