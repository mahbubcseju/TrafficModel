
def read_all_road_segments(file_name):
    f = open(file_name, "r")

    def check(s):
        s = s.strip()
        for ch in s:
            if ch.isdigit() or ch == ' ':
                continue
            else:
                return False
        return True

    road_counter, total, length, height = 0, 0, 0, 0
    co_ordinates, co_ordinates_st_en = [], []
    cnt, head_cnt, road_num = 0, 0, -1

    for line in f:
        line = line.strip('\n').strip(',')
        if len(line) <= 20 and check(line) and not head_cnt:
            a = line.split()
            if len(a) != 2:
                continue
            length = max(length, int(a[0]))
            height = max(height, int(a[1]))
            co_ordinates[road_num].append([int(a[0]) - 1, int(a[1]) - 1])
            total += 1
        elif len(line) >= 20 or head_cnt:
            if head_cnt == 0:
                head_cnt = 4
                road_num += 1
                co_ordinates.append([[line]])
            else:
                if head_cnt == 2:
                    start_points = line.strip().split()
                    co_ordinates_st_en.append([[start_points[1], start_points[2]]])
                elif head_cnt == 1:
                    start_points = line.strip().split()
                    co_ordinates_st_en[road_num].append([start_points[1], start_points[2]])
                head_cnt -= 1

    return co_ordinates_st_en, co_ordinates


def sort_coordinates(co_ordinates_st_en, co_ordinates):

    def processed_road_name(road_name):
        ans = ""
        cnt = 0
        for it in range(len(road_name)):
            if cnt >= 1:
                ans += road_name[it]
            if road_name[it] in ('(', ')'):
                cnt += 1

        st = 0
        for it in range(len(ans)):
            if ans[it] != ' ':
                st = it
                break
        en = len(ans)
        for it in range(len(ans) - 1, -1, -1):
            if ans[it] == ')':
                break
            en = it
        return str(ans[st:en])

    for road_id in range(len(co_ordinates)):
        co_ordinates[road_id][0][0] = processed_road_name(co_ordinates[road_id][0][0])
        for i in range(2):
            for j in range(2):
                co_ordinates_st_en[road_id][i][j] = int(co_ordinates_st_en[road_id][i][j]) - 1

        start_point = co_ordinates_st_en[road_id][0]
        end_point = co_ordinates_st_en[road_id][1]

        road_coordinates = co_ordinates[road_id][1:]
        start_ind = road_coordinates.index(start_point)
        road_coordinates[start_ind] = -1

        sorted_coordinates = [start_point]

        while road_coordinates.count(-1) != len(road_coordinates) and start_point != end_point:
            flag = False
            for i in range(len(road_coordinates)):
                if road_coordinates[i] != -1 and (abs(road_coordinates[i][0] - start_point[0]) + abs(
                        road_coordinates[i][1] - start_point[1])) == 1:
                    sorted_coordinates.append(road_coordinates[i])
                    start_point = road_coordinates[i]
                    road_coordinates[i] = -1
                    flag = True
                    break
            if flag:
                continue
            for i in range(len(road_coordinates)):
                if road_coordinates[i] != -1 and abs(road_coordinates[i][0] - start_point[0]) == 1 and abs(
                        road_coordinates[i][1] - start_point[1]) == 1:
                    sorted_coordinates.append(road_coordinates[i])
                    start_point = road_coordinates[i]
                    road_coordinates[i] = -1
                    flag = True
                    break
            if flag:
                continue

            ind = -1
            dis1 = 1000000000000000000
            for i in range(len(road_coordinates)):
                if road_coordinates[i] != -1:
                    dis = max((abs(road_coordinates[i][0] - start_point[0]), abs(
                        road_coordinates[i][1] - start_point[1])))
                    if dis1 > dis:
                        dis1 = dis
                        ind = i

            sorted_coordinates.append(road_coordinates[ind])
            start_point = road_coordinates[ind]
            road_coordinates[ind] = -1
            flag = 1
            continue
        co_ordinates[road_id][1:] = sorted_coordinates
    return co_ordinates


def get_all_road_segments(file_name):
    co_ordinates_start_end, co_ordinates_unsorted = read_all_road_segments(file_name)
    co_ordinates_sorted = sort_coordinates(co_ordinates_start_end, co_ordinates_unsorted)
    return co_ordinates_sorted
