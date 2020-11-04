import os
import csv


def make_expected_file(roads, images, continuous_list=[4], is_continuous=True, ignored_pixel=0):
    final = []

    header = ['Intersection']
    for j in range(len(images[0])):
        header.append(images[0][j])
    final.append(header)

    road_cnt = 0
    for ind in range(len(images)):
        row = images[ind]
        if len(row[0]) > 20:
            pros_road = row[0].replace(' ', '').lower()
            road_cnt += 1
            temp = [pros_road]
            for j in range(1, len(row)):
                temp.append(0)
            final.append(temp)

            if is_continuous:
                for j in range(1, len(row)):
                    cnt1, ignore, last = 0, 0, 0
                    for ind1 in range(ind + 1, len(images)):
                        if len(images[ind1][0]) > 20:
                            break
                        if int(images[ind1][j]) in continuous_list:
                            cnt1 += 1
                            ignore = 0
                        else:
                            ignore += 1
                        if ignore == ignored_pixel:
                            last = cnt1
                            cnt1 = 0
                            ignore = 0
                    final[road_cnt][j] += max(last, cnt1)

        elif not is_continuous:
            for j in range(1, len(row)):
                if int(row[j]) in continuous_list:
                    final[road_cnt][j] += 1

    return final


def process_data_for_regression(data, in_out, continous_list=None, is_continuous=True, ignored_pixel=0):
    return make_expected_file(in_out, data, continous_list, is_continuous, ignored_pixel)
