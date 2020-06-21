import os
import csv


def make_expected_file(roads, images, continuous_list=[4], is_continuous=True, ignored_pixel=0):
    final = []

    header = ['Intersection']
    for j in range(len(images[0])):
        header.append(images[0][j])
    final.append(header)

    road_intersection_index = {}

    for j in range(1, len(roads)):
        if int(roads[j][0]) == 1:
            continue
        road_intersection_index[roads[j][3].replace(' ', '').lower()] = int(roads[j][1])
        if roads[j][1] != roads[j-1][1]:
            final.append([roads[j][2]])
            for i in range(len(images[0])):
                final[int(roads[j][1])].append(0)

    cur_intersection = -1

    for ind in range(len(images)):
        row = images[ind]
        if len(row[0]) > 20:
            pros_road = row[0].replace(' ', '').lower()
            if pros_road in road_intersection_index:
                cur_intersection = road_intersection_index[
                    pros_road
                ]
                if is_continuous:
                    for j in range(1, len(row)):
                        cnt, ignore = 0, 0
                        for ind1 in range(ind + 1, len(images)):
                            if len(images[ind1][0]) > 20:
                                break
                            if int(images[ind1][j]) in continuous_list:
                                cnt += 1
                                ignore = 0
                            else:
                                ignore += 1
                            if ignore == ignored_pixel:
                                cnt = 0
                                ignore = 0
                        final[cur_intersection][j] += cnt
            else:
                cur_intersection = -1
        elif cur_intersection != -1 and not is_continuous:
            for j in range(1, len(row)):
                if int(row[j]) in continuous_list:
                    final[cur_intersection][j] += 1

    return final


def process_data_for_regression(data, in_out, continous_list=None, is_continuous=True, ignored_pixel=0):
    return make_expected_file(in_out, data, continous_list, is_continuous, ignored_pixel)
