#!/usr/bin/env python3
import os
import csv

path = "/home/kiyooka/catkin_ws/src/nav_cloning/data/analysis/robot_move/"
list_count = []
list_x = []
list_y = []
list_count2 = []
list_reset_count = []
num = 0
score = 0
before_reset_count = 0
before_count2 = 0
with open(path + 'traceable_pos.csv', 'r') as f:
    for row in csv.reader(f):
        str_count, str_x, str_y, str_the, str_traceable = row
        count, x, y = int(str_count), float(str_x), float(str_y)
        if count %7 != 4:
            list_count.append(count)
            list_x.append(x)
            list_y.append(y)


# with open(path + '300_use_dl_output/trajectory.csv', 'r') as f:
with open(path + '300_follow_line/trajectory.csv', 'r') as f:
# with open(path + 'gomi/trajectory.csv', 'r') as f:
# with open(path + 'use_dl_output/test/trajectory.csv', 'r') as f:
    for row in csv.reader(f):
        str_count2, str_x2, str_y2, str_reset_count = row
        count2, reset_count = int(str_count2), int(str_reset_count)
        list_count2.append(count2)
        list_reset_count.append(reset_count)


# with open(path + 'use_dl_output/test/traceable_pos.csv', 'w') as fw:
# with open(path + 'follow_line/test/traceable_pos.csv', 'w') as fw:
# with open(path + '300_use_dl_output/test/traceable_pos.csv', 'w') as fw:
with open(path + '300_follow_line/test/traceable_pos.csv', 'w') as fw:
# with open(path + 'gomi/test/traceable_pos.csv', 'w') as fw:
    writer= csv.writer(fw,lineterminator = '\n')
    for x,y in zip(list_count2,list_reset_count):
        if y != before_reset_count:
            # if before_count2 != 150:
            if before_count2 != 300:
                score += 1
            else:
                pass
            if y %3 == 0:
                num += 1
                score = score / 3.0
                line = [str(list_x[num-1]), str(list_y[num-1]), str(score)]
                writer.writerow(line)
                score = 0
            else:
                pass
        before_reset_count = y
        before_count2 = x
print(len(list_x))


