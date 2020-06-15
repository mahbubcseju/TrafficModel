import os
import csv

import pandas as pd
import numpy as np

from config import config
from make_csv_for_regression import process_data_for_regression
from utils.df_to_list import df_to_list, csv_to_list


current_directory = os.getcwd()
train, test = None, None


for key in config['train']:
    value = config['train'][key]
    start_date = value['start_date'] if 'start_date' in value else '01_11_2019'
    end_date = value['end_date'] if 'end_date' in value else '01_11_2019'
    start_time = value['start_time'] if 'start_time' in value else '06_00_01'
    end_time = value['end_time'] if 'end_time' in value else '23_59_31'

    start_day, end_day = int(start_date.split('_')[0]), int(end_date.split('_')[0])
    data, flag = None,  None
    for day in range(start_day, end_day + 1):
        day_div = (day - 1) // 5
        start_time_stamp = '{:02d}_11_2019 {}'.format(day, start_time)
        end_time_stanp = '{:02d}_11_2019 {}'.format(day, end_time)

        if (day - 1) % 5 == 0 or not flag:
            data_set = 'november{}.csv'.format(day_div)
            data_path = os.path.join(current_directory, 'csvs', data_set)
            data = pd.read_csv(data_path, index_col=0, low_memory=False)

        flag = True
        current_data = data.loc[:, start_time_stamp:end_time_stanp]
        train = pd.concat([train, current_data], axis=1)


for key in config['test']:
    value = config['test'][key]
    start_date = value['start_date'] if 'start_date' in value else '01_11_2019'
    end_date = value['end_date'] if 'end_date' in value else '01_11_2019'
    start_time = value['start_time'] if 'start_time' in value else '06_00_01'
    end_time = value['end_time'] if 'end_time' in value else '23_59_31'

    start_day, end_day = int(start_date.split('_')[0]), int(end_date.split('_')[0])
    data, flag = None, None
    for day in range(start_day, end_day + 1):
        day_div = (day - 1) // 5
        start_time_stamp = '{:02d}_11_2019 {}'.format(day, start_time)
        end_time_stanp = '{:02d}_11_2019 {}'.format(day, end_time)

        if (day - 1) % 5 == 0 or not flag:
            data_set = 'november{}.csv'.format(day_div)
            data_path = os.path.join(current_directory, 'csvs', data_set)
            data = pd.read_csv(data_path, index_col=0, low_memory=False)

        flag = True
        current_data = data.loc[:, start_time_stamp:end_time_stanp]
        test = pd.concat([test, current_data], axis=1)

in_out_data = csv_to_list(
    current_directory,
    'updated_IntersectionsWithIncomingOutgoing_Mirpur.csv'
)

intensity_list_to_count = value['intensity_list_to_count'] if 'intensity_list_to_count' in value else [4]
is_intensity_continuous = value['is_intensity_continuous'] if 'is_intensity_continuous' in value else True
number_of_ignored_cell = value['number_of_ignored_cell'] if 'number_of_ignored_cell' in value else 1

train_data = process_data_for_regression(
    df_to_list(train),
    in_out_data,
    continous_list=intensity_list_to_count,
    is_continuous=is_intensity_continuous,
    ignored_pixel=number_of_ignored_cell,
)

test_data = process_data_for_regression(
    df_to_list(test),
    in_out_data,
    continous_list=intensity_list_to_count,
    is_continuous=is_intensity_continuous,
    ignored_pixel=number_of_ignored_cell,
)

sampling_rate = value['sampling_rate'] if 'sampling_rate' in value else 2
seq_len = value['seq_len'] if 'seq_len' in value else  60
pre_len = value['pre_len'] if 'pre_len' in value else 10
repeat =  value['repeat'] if 'repeat' in value else False
is_continuous =  value['is_continous'] if 'is_continous' in value else False

print(train_data)

