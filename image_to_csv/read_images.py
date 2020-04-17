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


def read_images(path, co_ordinates):
    total_image = 0
    rows = []
    for i in range(len(co_ordinates)):
        for col in co_ordinates[i]:
            if len(col) == 1:
                rows.append([col[0]])
            else:
                rows.append([' '.join([str(j) for j in col])])

    for file in sorted(listdir(path)):
        if not isfile(join(path, file)) or file.split('.')[1] != 'png':
            continue
        total_image += 1
        image = imread(join(path, file))

        co = 0
        for i in range(len(co_ordinates)):
            for col in co_ordinates[i]:
                if len(col) == 1:
                    rows[co].append(str(file.split('.')[0]))
                else:
                    rows[co].append(color_map[color_value(image[col[0]][col[1]][:3])])
                co += 1
    return rows
