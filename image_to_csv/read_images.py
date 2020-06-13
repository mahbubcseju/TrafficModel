import numpy as np
from os import listdir
from os.path import isfile, join

from skimage.io import imread

color_map = {
    'Green': 1,
    'Orange': 2,
    'Red': 3,
    'Dark Red': 4,
    'Default': 0,
}


def color_value(colors):
    r, g, b = map(int, colors)
    if (132 <= r <= 210) and (211 <= g <= 235) and (117 <= b <= 205):
        return 'Green'
    elif (237 <= r <= 246) and (155 <= g <= 225) and (91 <= b <= 205):
        return 'Orange'
    elif (220 <= r <= 239) and (76 <= g <= 202) and (60 <= b <= 197):
        return 'Red'
    elif (117 <= r <= 176) and (37 <= g <= 134) and (33 <= b <= 133):
        return 'Dark Red'
    else:
        return 'Default'


def create_file_list(_range):
    st_date = _range['start_date']
    en_date = _range['end_date']
    st_sp = st_date.split('_')
    en_sp = en_date.split('_')

    file_list = []
    for each_day in range(int(st_sp[0]), int(en_sp[0]) + 1):
        cur_date = '{:02d}_{}_{}'.format(each_day, st_sp[1], st_sp[2])
        for each_hour in range(6, 24):
            for each_minute in range(60):
                for seconds in (1, 31):
                    cur_file_name = '{} {:02d}_{:02d}_{:02d}.png'.format(cur_date, each_hour, each_minute, seconds)
                    file_list.append(cur_file_name)

    return file_list


def read_images(path, _range, co_ordinates):
    total_image = 0
    rows = []
    for i in range(len(co_ordinates)):
        for col in co_ordinates[i]:
            if len(col) == 1:
                rows.append([col[0]])
            else:
                rows.append([' '.join([str(j) for j in col])])

    file_list = create_file_list(_range)

    em_image = np.array([[[0] * 4 for j in range(1600)] for i in range(1200)])
    for fil in file_list:
        print(fil)
        total_image += 1
        try:
            image = imread(join(path, fil))
        except Exception as e:
            image = em_image

        co = 0
        for i in range(len(co_ordinates)):
            for col in co_ordinates[i]:
                if len(col) == 1:
                    rows[co].append(str(fil.split('.')[0]))
                else:
                    rows[co].append(color_map[color_value(image[col[0]][col[1]][:3])])
                co += 1
        if total_image % 100 == 0:
            print('Total Image', total_image)
    print(total_image)
    return rows
