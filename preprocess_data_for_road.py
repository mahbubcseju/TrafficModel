import os

from image_to_csv import convert_image_to_csv
from make_csv_for_road_intensity import make_csv_for_regression

current_directory = os.getcwd()
_range = {
    'start_date': '26_11_2019',
    'end_date': '30_11_2019',
}
file = convert_image_to_csv(current_directory, 'november', _range, 'mirpurAllRoadSegmentsUpdated.csv')

file_for_regression = make_csv_for_regression(
    current_directory,
    file,
    [3, 4],
)

print(file_for_regression)
