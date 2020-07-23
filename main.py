import os
import csv

import pandas as pd
import numpy as np

from config import config
from make_csv_for_regression import process_data_for_regression
from utils.df_to_list import df_to_list, csv_to_list
from utils import save_intermediate_file


current_directory = os.getcwd()


def train_generator():
    for key in config['train']:
        value = config['train'][key]
        start_date = value['start_date'] if 'start_date' in value else '01_11_2019'
        end_date = value['end_date'] if 'end_date' in value else '01_11_2019'
        start_time = value['start_time'] if 'start_time' in value else '06_00_01'
        end_time = value['end_time'] if 'end_time' in value else '23_59_31'

        start_day, end_day = int(start_date.split('_')[0]), int(end_date.split('_')[0])
        data, flag, train_cur = None,  None, None
        for day in range(start_day, end_day + 1):
            day_div = (day - 1) // 5
            start_time_stamp = '{:02d}_11_2019 {}'.format(day, start_time)
            end_time_stamp = '{:02d}_11_2019 {}'.format(day, end_time)

            if (day - 1) % 5 == 0 or not flag:
                data_set = 'november{}.csv'.format(day_div)
                data_path = os.path.join(current_directory, 'csvs', data_set)
                data = pd.read_csv(data_path, index_col=0, low_memory=False)

            flag = True
            current_data = data.loc[:, start_time_stamp:end_time_stamp]
            # train_cur = pd.concat([train_cur, current_data], axis=1)
            print('succesffully read train data of day {} '.format(day))
            yield current_data


def test_generator():
    for key in config['test']:
        value = config['test'][key]
        start_date = value['start_date'] if 'start_date' in value else '01_11_2019'
        end_date = value['end_date'] if 'end_date' in value else '01_11_2019'
        start_time = value['start_time'] if 'start_time' in value else '06_00_01'
        end_time = value['end_time'] if 'end_time' in value else '23_59_31'

        start_day, end_day = int(start_date.split('_')[0]), int(end_date.split('_')[0])
        data, flag, test_cur = None, None, None
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
            # test_cur = pd.concat([test_cur, current_data], axis=1)
            print('succesffully read test data of day {} '.format(day))
            yield current_data


in_out_data = csv_to_list(
    current_directory,
    'updated_IntersectionsWithIncomingOutgoing_Mirpur.csv'
)

intensity_list_to_count = config['intensity_list_to_count'] if 'intensity_list_to_count' in config else [4]
is_intensity_continuous = config['is_intensity_continuous'] if 'is_intensity_continuous' in config else True
number_of_ignored_cell = config['number_of_ignored_cell'] if 'number_of_ignored_cell' in config else 1
intermediate_file_save = config['intermediate_file_save'] if 'intermediate_file_save' in config else False


train, test = [], []

for train1 in train_generator():
    train_data = process_data_for_regression(
        df_to_list(train1),
        in_out_data,
        continous_list=intensity_list_to_count,
        is_continuous=is_intensity_continuous,
        ignored_pixel=number_of_ignored_cell,
    )
    train.append(train_data)

if intermediate_file_save:
    save_intermediate_file(current_directory, train, 'train')


for test1 in test_generator():
    test_data = process_data_for_regression(
        df_to_list(test1),
        in_out_data,
        continous_list=intensity_list_to_count,
        is_continuous=is_intensity_continuous,
        ignored_pixel=number_of_ignored_cell,
    )
    test.append(test_data)


if intermediate_file_save:
    save_intermediate_file(current_directory, test, 'test')
print('Successfully make in out data')


sampling_rate = config['sampling_rate'] if 'sampling_rate' in config else 2
seq_len = config['seq_len'] if 'seq_len' in config else  60
pre_len = config['pre_len'] if 'pre_len' in config else 10
repeat = config['repeat'] if 'repeat' in config else False
is_continuous = config['is_continous'] if 'is_continous' in config else False


from run_models import run_models

result = [['Sampling Rate', 'Sequence length', 'Predicted length', "", 'RMSE', 'MAE', 'COR', 'R2', 'invalid/total']]

ans = run_models(
    current_directory,
    train,
    test,
    sampling_rate=sampling_rate,
    seq_len=seq_len,
    pre_len=pre_len,
    repeat=repeat,
    is_continuous=is_continuous,
)
result += ans

# sam = [1, 2, 10]
sam = [1]
seq = [15, 30, 45, 60]
pre = [5, 15, 30, 45, 60]
# pre = [5, 15, 30, 45, 60]

import threading


def run_models_within_threading(cd, tr, tt, sr=1, sl=12, pl=3, rt=False, ic=True, rst=None):
    ans1 = run_models(
        cd,
        tr,
        tt,
        sampling_rate=sr,
        seq_len=sl,
        pre_len=pl,
        repeat=rt,
        is_continuous=ic,
    )
    rst += ans1


for sa in sam:
    for se in seq:
        for pr in pre:
            thread = threading.Thread(
                target=run_models_within_threading,
                args=(
                    current_directory,
                    train,
                    test,
                    sa,
                    (se * 2)//sa,
                    (pre_len * 2)//sa,
                    repeat,
                    is_continuous,
                    result)
            )
            thread.start()

# final_file = os.path.join(current_directory, 'csvs', 'table.csv')
# with open(final_file, 'w') as writer:
#     wr = csv.writer(writer)
#     wr.writerows(result)
#
# print('FINAL COMPLETE')
