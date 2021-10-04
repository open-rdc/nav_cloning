#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import csv
import math
import numpy as np

def calc_histogram_training_posture():
    rospy.init_node('calc_histogram_training_posture_node', anonymous=True)
    path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
    division = 10
    with open(path + 'path.csv', 'r') as f:
        is_first = True
        points=[]
        x0, y0 = 0.0, 0.0
        for row in csv.reader(f):
            if is_first:
                is_first = False
                continue
            str_path_no, str_x, str_y = row
            x, y = float(str_x), float(str_y)
            print("* "+str(x)+", "+str(y))
            for i in range(division):
                points.append([(x-x0)*i/division+x0, (y-y0)*i/division+y0])
                print(str((x-x0)*i/division+x0)+", "+str((y-y0)*i/division+y0))
            x0, y0 = x, y

    with open(path +  'histogram_training_posture.csv', 'w') as fw:
        writer = csv.writer(fw, lineterminator='\n')
        with open(path + 'training.csv', 'r') as f:
            is_first = True
            for row in csv.reader(f):
                if is_first:
                    is_first = False
                    continue
                str_step, str_mode, str_loss, str_angle_error, str_distance, str_x, str_y, str_the = row
                x, y, the = float(str_x), float(str_y), float(str_the)
                min_len_sq = 10000.0
                min_no = 0
                for i in range(len(points)-10):
                    p = points[i]
                    l_sq = (x - p[0])**2+(y - p[1])**2
                    if l_sq < min_len_sq:
                        min_len_sq = l_sq
                        min_no = i
                print(str(min_no)+"/"+str(len(points)))
                min_len = math.sqrt(min_len_sq)
                diff_ang = the - math.atan2(points[min_no+10][1]-points[min_no][1], points[min_no+10][0]-points[min_no][0])
                diff_ang = diff_ang - 2 * math.pi if diff_ang >  math.pi else diff_ang 
                diff_ang = diff_ang + 2 * math.pi if diff_ang < -math.pi else diff_ang 
                line = [str(min_len), str(diff_ang)]
                writer.writerow(line)
                
if __name__ == '__main__':
    calc_histogram_training_posture()

