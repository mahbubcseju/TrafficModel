import os
import csv


def make_expected_file(roads, images, value, path, is_continuous):
    final = []

    header = ['roads']
    for j in range(1, len(images[0])):
        header.append(images[0][j])
    final.append(header)

    road_index = {}

    for j in range(1, len(roads)):
        if int(roads[j][0]) == 1:
            continue
        road_index[roads[j][3].replace(' ', '').lower()] = j
        final.append([roads[j][3]])
        for i in range(1, len(images[0])):
            final[j].append(0)

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


def make_csv_for_regression(base_directory, road_segments_csv, in_out_csv, continous_list=None, is_continuous=True):
    absoulte_in_out_csv = os.path.join(base_directory, 'csvs', in_out_csv)
    absoulte_road_segments_csv = os.path.join(base_directory, 'csvs', road_segments_csv)
    final_path = os.path.join(base_directory, 'csvs', str(road_segments_csv.split('.')[0]) + 'in_out.csv')

    roads_info = []
    with open(absoulte_in_out_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for rows in reader:
            roads_info.append(rows)

    image_info = []
    with open(absoulte_road_segments_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for rows in reader:
            image_info.append(rows)


    return make_expected_file(roads_info, image_info, continous_list, final_path, is_continuous)
