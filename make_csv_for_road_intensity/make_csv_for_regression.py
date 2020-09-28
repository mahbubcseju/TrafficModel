import os
import csv


def make_expected_file(images, value, path):
    final = []

    header = ['roads']
    for j in range(1, len(images[0])):
        header.append(images[0][j])
    final.append(header)

    road_index = {}
    cnt = 1
    for ind in range(len(images)):
        row = images[ind]
        if len(row[0]) > 20:
            pros_road = row[0].replace(' ', '').lower()
            road_index[pros_road] = cnt
            final.append([row[0]])
            for i in range(1, len(images[0])):
                final[cnt].append(0)
            cnt = cnt + 1

    cur_intersection = -1
    for ind in range(len(images)):
        row = images[ind]
        if len(row[0]) > 20:
            pros_road = row[0].replace(' ', '').lower()
            if pros_road in road_index:
                cur_intersection = road_index[
                    pros_road
                ]
            else:
                cur_intersection = -1
        elif cur_intersection != -1:
            for j in range(1, len(row)):
                if int(row[j]) in value:
                    final[cur_intersection][j] += 1

    with open(path, 'w') as writer:
        wr = csv.writer(writer)
        wr.writerows(final)
    return path


def make_csv_for_regression(base_directory, road_segments_csv, continous_list=None, is_continuous=True):
    absoulte_road_segments_csv = os.path.join(base_directory, 'csvs', road_segments_csv)
    final_path = os.path.join(base_directory, 'csvs', str(road_segments_csv.split('.')[0]) + '_roads.csv')

    image_info = []
    with open(absoulte_road_segments_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for rows in reader:
            image_info.append(rows)

    return make_expected_file(image_info, continous_list, final_path)
