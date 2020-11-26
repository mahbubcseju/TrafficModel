import os
import csv
import time
import pandas as pd
import numpy as np

from config import config
from make_csv_for_regression import process_data_for_regression
from utils.df_to_list import df_to_list, csv_to_list
from utils import save_intermediate_file
from utils import preprocess_data
from utils import process_per_segment

from models import ha_graph, svr_graph, arima_graph

current_directory = os.getcwd()
base_directory = current_directory

dataset = 'new_road.csv'
method = 'mean'
sequence_length = 15
prediction_length = 3

data_path = os.path.join(current_directory, 'csvs', dataset)

df = pd.read_csv(data_path)
df = df.set_index('Intersection')
df = df.T

df.index = pd.to_datetime(df.index, format='%d_%m_%Y %H_%M_%S')


if method == 'mean':
    data = df.resample('5T').mean().dropna()
elif method == 'max':
    data = df.resample('5T').max().dropna()
elif method == 'min':
    data = df.resample('5T').min().dropna()
else:
    data = df.resample('5T').median.dropna()

train = data.loc[:'2019-11-20 23:59:31']
test = data.loc['2019-11-21 06:00:00':]


headers = train.columns

# ha(data, pre_len=1, repeat=False, is_continuous=True)
result = [['', ''],['Node Number', 'Node name']]
ans = []


be_ha = time.time()
ha_header, ha_test, ha_result = ha_graph(train, test, sequence_length=sequence_length, prediction_length=prediction_length)
print(len(ha_header))
for i in range(len(ha_header)):
    result.append([i, ha_header[i]])
result.append(['Average', ''])
result = np.array(result)

print(len(result))
ha_temp_result = process_per_segment('HA', ha_test, ha_result)
print(len(ha_temp_result))
result = np.concatenate([result, ha_temp_result], axis=1)
ha_avg = ['HA']
for i in ha_temp_result[-1]:
    ha_avg.append(i)
ans.append(ha_avg)

print('HA Complete')

be_svr = time.time()
svr_header, svr_test, svr_result = svr_graph(train, test, sequence_length=sequence_length, prediction_length=prediction_length)
svr_temp_result = process_per_segment('SVR', svr_test, svr_result)
result = np.concatenate([result, svr_temp_result], axis=1)

svr_avg = ['SVR']
for i in svr_temp_result[-1]:
    svr_avg.append(i)
ans.append(svr_avg)

print('SVR Complete')




be_arima = time.time()
arima_header, arima_test, arima_result, count_invalid, total = arima_graph(train, test,sequence_length=sequence_length, prediction_length=prediction_length, p=1, d=0, q=0)
arima_temp_result = process_per_segment('ARIMA', arima_test, arima_result)
result = np.concatenate([result, arima_temp_result], axis=1)
_avg = [ 'ARIMA']
for i in arima_temp_result[-1]:
    _avg.append(i)
_avg.append(str(count_invalid) + "/" + str(total))
ans.append(_avg)
print('Arima complete')

af_arima = time.time()

final_result = []

file_name = 'result_road.csv'
image_data_csv_file = os.path.join(base_directory, 'csvs', file_name )

with open(image_data_csv_file, 'w') as writer:
    wr = csv.writer(writer)
    wr.writerows(result)

print('FINAL COMPLETE')
print("Ha Takes: ", be_svr - be_ha)
print("SVR_Graph Takes", be_arima - be_svr)
print("Arima takes", af_arima - be_arima)

print(ans)
