import csv
import os

from .get_road_segment import get_all_road_segments
from .read_images import read_images


def convert_image_to_csv(base_directory, image_directory, _range, road_segments_filename):

    absolute_image_directory = os.path.join(base_directory, image_directory)
    absolute_segments_file = os.path.join(base_directory, 'csvs', road_segments_filename)
    image_data_csv_file = os.path.join(base_directory, 'csvs', str(image_directory) + '.csv')

    co_ordinates = get_all_road_segments(absolute_segments_file)
    image_data = read_images(absolute_image_directory, _range, co_ordinates)

    with open(image_data_csv_file, 'w') as writer:
        wr = csv.writer(writer)
        wr.writerows(image_data)

    return image_data_csv_file
