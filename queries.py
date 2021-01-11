import os
import csv
import time
import pandas as pd
import math
import numpy as np

from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

# Configurations
prediction_length = 9
start_time = '08:00'
end_time = '09:00'

##############################


current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'csvs')
models = ['HA', 'SVR', 'SVR_GRAPH', 'ARIMA']


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for ind in range(len(y_true)):
        if y_true[ind] == 0:
            y_true[ind] += 1
            y_pred[ind] + 1
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calc_errors(ground_truth, predicted_value):
    rmse = math.sqrt(mean_squared_error(ground_truth, predicted_value))
    mae = mean_absolute_error(ground_truth, predicted_value)
    mape = mean_absolute_percentage_error(ground_truth, predicted_value)
    return rmse, mae, mape


for model in models:
    file_path = os.path.join(data_path, f'{model}_{5 * prediction_length}.csv')
    data = pd.read_csv(file_path)
    data = data.set_index('time')
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
    data = data['2019-11-24 12:05:00':]
    data['time_only'] = data.index.map(lambda x: str(x).split()[1])
    filtered_data = data.loc[(start_time <= data['time_only']) & (data['time_only'] <= end_time)]
    filtered_data = filtered_data.set_index('time_only')
    values = filtered_data.values.flatten()
    ground_truth = list(map(lambda x: float(x.split('X')[0]), values))
    predicted_value = list(map(lambda x: float(x.split('X')[1]), values))
    rmse, mae, mape = calc_errors(ground_truth, predicted_value)
    print(f'{model} Predicion after {prediction_length * 5} minutes: RMSE = {rmse}, MAE = {mae}, MAPE = {mape}')