#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import csv
import math
import numpy as np

def calc_traceable_histogram():
    rospy.init_node('calc_traceable_histogram_node', anonymous=True)
    path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
    offset_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    sum_of_traceable = [0.0] * len(offset_y)
    num_of_data = 0
    i = 0
    with open(path + 'traceable_pos.csv', 'r') as f:
        for row in csv.reader(f):
            str_x, str_y, str_the, str_traceable = row
            traceable = float(str_traceable)
            if traceable == 1.0:
               sum_of_traceable[i] += traceable
            i += 1
            if i >= len(offset_y):
                i = 0
                num_of_data += 1

    with open(path +  'traceable_histogram.csv', 'w') as fw:
        writer = csv.writer(fw, lineterminator='\n')
        for i in range(len(offset_y)):
            ave_of_traceable = sum_of_traceable[i]/num_of_data
            line = [str(offset_y[i]), str(ave_of_traceable)]
            writer.writerow(line)

if __name__ == '__main__':
    calc_traceable_histogram()

