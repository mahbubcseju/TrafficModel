import os
import csv

def df_to_list(df):
    return [
               df.columns.tolist()
           ] + df.reset_index().values.tolist()

def csv_to_list(base_directory, in_out_csv):
    absoulte_in_out_csv = os.path.join(base_directory, 'csvs', in_out_csv)

    roads_info = []
    with open(absoulte_in_out_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for rows in reader:
            roads_info.append(rows)

    return roads_info