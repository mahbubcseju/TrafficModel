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
from utils import process_per_segment, write_results
from models import ha_graph, svr_graph, arima_graph, svr_graph_graph

current_directory = os.getcwd()
base_directory = current_directory

sampling_rate = '5T'
dataset = 'new_road.csv'
method = 'mean'
sequence_length = 12
prediction_length = 9

data_path = os.path.join(current_directory, 'csvs', dataset)

df = pd.read_csv(data_path)
df = df.set_index('Intersection')
df = df.T

df.index = pd.to_datetime(df.index, format='%d_%m_%Y %H_%M_%S')


if method == 'mean':
    data = df.resample(sampling_rate).mean().dropna()
elif method == 'max':
    data = df.resample(sampling_rate).max().dropna()
elif method == 'min':
    data = df.resample(sampling_rate).min().dropna()
else:
    data = df.resample(sampling_rate).median.dropna()


train = data.loc[:'2019-11-15 23:59:31']
test = data.loc['2019-11-24 06:00:00':]


predicted_time_stamp = test.index[sequence_length + prediction_length:]

headers = train.columns

# ha(data, pre_len=1, repeat=False, is_continuous=True)
path = os.path.join(current_directory, 'csvs')
result = [['', ''], ['Node Number', 'Node name']]
ans = []


be_ha = time.time()
ha_header, ha_test, ha_result = ha_graph(train, test, sequence_length=sequence_length, prediction_length=prediction_length)
write_results(predicted_time_stamp, ha_header, ha_test, ha_result,  f'{path}/HA_{5 * prediction_length}.csv')

for i in range(len(ha_header)):
    result.append([i, ha_header[i]])
result.append(['Average', ''])
result = np.array(result)

ha_temp_result = process_per_segment('HA', ha_test, ha_result)
result = np.concatenate([result, ha_temp_result], axis=1)
ha_avg = ['HA']
for i in ha_temp_result[-1]:
    ha_avg.append(i)
ans.append(ha_avg)
print('HA Complete')

be_svr = time.time()
svr_header, svr_test, svr_result = svr_graph(train, test, sequence_length=sequence_length, prediction_length=prediction_length)
write_results(predicted_time_stamp, svr_header, svr_test, svr_result,  f'{path}/SVR_{5 * prediction_length}.csv')

svr_temp_result = process_per_segment('SVR', svr_test, svr_result)
result = np.concatenate([result, svr_temp_result], axis=1)

svr_avg = ['SVR']
for i in svr_temp_result[-1]:
    svr_avg.append(i)
ans.append(svr_avg)

print('SVR Complete')
# adjacent_path = os.path.join('csvs', 'adjacencyMatrixUpdated.csv')
# adjacency_matrix = pd.read_csv(adjacent_path, index_col=0)

be_graph = time.time()
svr_graph_header, svr_graph_test, svr_graph_result = svr_graph_graph(train, test, sequence_length=sequence_length, prediction_length=prediction_length)
write_results(predicted_time_stamp, svr_graph_header, svr_graph_test, svr_graph_result,  f'{path}/SVR_GRAPH_{5 * prediction_length}.csv')
svr_tempg_result = process_per_segment('SVR', svr_graph_test, svr_graph_result)
result = np.concatenate([result, svr_tempg_result], axis=1)

svr_avg = ['SVR']
for i in svr_tempg_result[-1]:
    svr_avg.append(i)
ans.append(svr_avg)
print('SVR_GRAPH Complete')

be_arima = time.time()
arima_header, arima_test, arima_result, count_invalid, total = arima_graph(train, test,sequence_length=sequence_length, prediction_length=prediction_length, p=1, d=0, q=0)
write_results(predicted_time_stamp, arima_header, arima_test, arima_result,  f'{path}/ARIMA_{5 * prediction_length}.csv')
arima_temp_result = process_per_segment('ARIMA', arima_test, arima_result)
result = np.concatenate([result, arima_temp_result], axis=1)
_avg = ['ARIMA']
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
print("Graph Takes", be_graph - be_svr)
print("SVR_Graph Takes", be_arima - be_graph)
print("Arima takes", af_arima - be_arima)

print(ans)
