import os
import csv
import datetime

import pandas as pd
import numpy as np


def save_intermediate_file(current_dir, data, file_prefix):
    final_data = None
    for sub_data in data:
        np_data = np.array(sub_data)
        final_data = np_data if not final_data else np.concatenate([final_data, np_data[:, 1:]], axis=1)

    file_name_relative = '{}_{}_inout.csv'.format(
        file_prefix,
        datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    )
    file_name_abs = os.path.join(current_dir, 'csvs', file_name_relative)
    with open(file_name_abs, 'w') as writer:
        wr = csv.writer(writer)
        wr.writerows(final_data)

    return file_name_abs
