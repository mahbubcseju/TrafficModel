import os

from image_to_csv import convert_image_to_csv
from .make_csv_for_regression import make_csv_for_regression

current_directory = os.getcwd()
file = convert_image_to_csv(current_directory, '13-11-19_mirpur', 'mirpurAllRoadSegments.csv')

file_for_regression = make_csv_for_regression(
    current_directory,
    file,
    'IntersectionsWithIncomingOutgoing_Mirpur.csv',
    [4],
    True,
)

print(file_for_regression)
