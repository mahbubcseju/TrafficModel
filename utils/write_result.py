import csv
import numpy as np


def write_results(time_stamp, header, test, result, file_name):

    ans =[['time']]
    for time_ in list(time_stamp):
        ans.append([time_])

    for i, val in enumerate(header):
        actual = test[i]
        predicted = result[i]
        node = header[i]
        assert (len(actual) == len(predicted))
        assert (len(time_stamp) == len(actual))
        ans[0].append(node)
        for j in range(len(actual)):
            ans[j+1].append(f'{actual[j]}X{predicted[j]}')

    with open(file_name, 'w') as writer:
        wr = csv.writer(writer)
        wr.writerows(ans)
