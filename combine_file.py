import os
import pandas as pd

base_dir = os.getcwd()
file_dir = os.path.join(base_dir, "csvs")


table = [["Sampling Rate", "Sequence Length(min)", "Predicted Length(min)", "Method", "RMSE", "MAE", "COR", "R2"]]

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
    reader = open(cur_file, "r")
    last = None
    for row in reader:
        last = row

    ext = file.split(".")
    para = ext[0].split("-")[1:6:2]
    para = list(map(int, para))
    para[1], para[2] = (para[1] * para[0])//2, (para[2] * para[0]) //2
    last = [el.replace("\n", "")for el in last.split(",")[2:]]

    method = ['HA', 'SVR', 'SVR_GRAPH', 'ARIMA']
    cnt = 0
    for k in range(0, 16, 4):
        ans = [para[0], para[1], para[2], method[cnt], last[k], last[k +1], last[k +2], last[k + 3]]
        cnt += 1
        table.append(ans)


import csv
image_data_csv_file = os.path.join(base_dir, "csvs", 'table.csv')
with open(image_data_csv_file, 'w') as writer:
    wr = csv.writer(writer)
    wr.writerows(table)
