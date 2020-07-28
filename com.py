import os
import pandas as pd

base_dir = os.getcwd()
file_dir = os.path.join(base_dir, "csvs")


table = [["Sampling Rate", "Sequence Length(min)", "Predicted Length(min)","HA RMSE", "HA MAE", "HA Cor", "HA R2", "SVR RMSE",
          "SVR MAE", "SVR COR", "SVR R2", "SVR GRAPH RMSE", "SVR GRAPH MAE", "SVR GRAPH Cor", "SVR GRAPH R2", "ARIMA RMSE", "ARIMA MAE", "ARIMA Cor", "ARIMA R2"]]

files = []
for file in os.listdir(file_dir):
    cur_file = os.path.join(file_dir, file)
    if os.path.isfile(cur_file):
        ext = file.split(".")
        if len(ext) < 2 or ext[1] != 'csv':
            continue
        if ext[0][0] != 'S':
            continue
        files.append(file)

files = sorted(files, key=lambda x: x.split("-")[1:6:2])

for file in files:
    cur_file = os.path.join(file_dir, file)
    if os.path.isfile(cur_file):
        ext = file.split(".")
        if len(ext) < 2 or ext[1] != 'csv':
            continue
        if ext[0][0] != 'S':
            continue

        reader = open(cur_file, "r")
        last = None
        for row in reader:
            last = row

        para = ext[0].split("-")[1:6:2]
        para = list(map(int, para))
        para[1], para[2] = (para[1] * para[0])//2, (para[2] * para[0]) //2
        last = [el.replace("\n", "")for el in last.split(",")[2:]]

        for k in last:
            para.append(k)

        table.append(para)


import csv
image_data_csv_file = os.path.join(base_dir, "csvs", 'table1.csv')
with open(image_data_csv_file, 'w') as writer:
    wr = csv.writer(writer)
    wr.writerows(table)
