import os
import csv

import pandas as pd
import numpy as np

from config import config
from make_csv_for_regression import process_data_for_regression
from utils.df_to_list import df_to_list, csv_to_list


current_directory = os.getcwd()
data_set = 'november{}.csv'.format(0)
data_path = os.path.join(current_directory, 'csvs', data_set)
data = pd.read_csv(data_path, index_col=0, low_memory=False)

current_data = data.loc[:, '01_11_2019 12_33_31':'01_11_2019 13_33_31']

data = df_to_list(current_data)

image_data_csv_file = os.path.join(current_directory, 'csvs', 'november_5.csv' )

with open(image_data_csv_file, 'w') as writer:
    wr = csv.writer(writer)
    wr.writerows(data)

print(current_data)
