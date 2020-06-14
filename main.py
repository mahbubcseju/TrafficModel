import os
import csv

import pandas as pd
import numpy as np

current_directory = os.getcwd()

from config import config


for key in config['train']:
    value = config['train'][key]
    start_date = value['start_date'] if 'start_date' in value else '01_11_2019'
    end_date = value['end_date'] if 'end_date' in value else '01_11_2019'
    start_time = value['start_time'] if 'start_time' in value else '06_00_01'
    end_time = value['end_time'] if 'end_time' in value else '23_59_31'

    start_day, end_day = int(start_date.split('_')[0]), int(end_date.split('_')[0])
    data, train, flag = None, None, None
    for day in range(start_day, end_day + 1):
        day_div = (day - 1) // 5
        start_time_stamp = '{:02d}_11_2019 {}'.format(day, start_time)
        end_time_stanp = '{:02d}_11_2019 {}'.format(day, end_time)

        if day % 5 == 0 or not flag:
            data_set = 'november{}.csv'.format(day // 5)
            data_path = os.path.join(current_directory, 'csvs', data_set)
            data = pd.read_csv(data_path, index_col=0, low_memory=False)

        flag = True

        current_data = data.loc[:, start_time_stamp:end_time_stanp]
        train = pd.concat([train, current_data], axis=1)
        print(train)


for key in config['test']:
    value = config['test'][key]
    start_date = value['start_date'] if 'start_date' in value else '01_11_2019'
    end_date = value['end_date'] if 'end_date' in value else '01_11_2019'
    start_time = value['start_time'] if 'start_time' in value else '06_00_01'
    end_time = value['end_time'] if 'end_time' in value else '23_59_31'

    start_day, end_day = int(start_date.split('_')[0]), int(end_date.split('_')[0])
    data, train, flag = None, None, None
    for day in range(start_day, end_day + 1):
        day_div = (day - 1) // 5
        start_time_stamp = '{:02d}_11_2019 {}'.format(day, start_time)
        end_time_stanp = '{:02d}_11_2019 {}'.format(day, end_time)

        if day % 5 == 0 or not flag:
            data_set = 'november{}.csv'.format(day // 5)
            data_path = os.path.join(current_directory, 'csvs', data_set)
            data = pd.read_csv(data_path, index_col=0, low_memory=False)

        flag = True

        current_data = data.loc[:, start_time_stamp:end_time_stanp]
        train = pd.concat([train, current_data], axis=1)
        print(train)

# import os
# import csv
#
# import pandas as pd
# import numpy as np
#
# current_directory = os.getcwd()
#
# data_set = 'november0.csv'
#
# data_path = os.path.join(current_directory, 'csvs', data_set)
# data = pd.read_csv(data_path, index_col=0)
#
# start_index = '01_11_2019 06_00_01'
# end_index = '01_11_2019 06_01_31'
# data = data.loc[:, start_index: end_index]
# print(data.columns.values)
# print(data.index)
# print(data)
