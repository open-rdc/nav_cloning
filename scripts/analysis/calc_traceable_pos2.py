#!/usr/bin/env python
from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../pytorch")
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import tf
import cv2
# from sensor_msgs.msg import Image
from nav_cloning_pytorch import *
import csv
import math

class calc_traceable_pos:
    def __init__(self):
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/reverse/'
        self.pos_no = 0
        self.save_img_no = 0
        self.path_points=[]
    def calc_traceable_pos(self):
        with open(self.path +  'path.csv', 'r') as f:
            division = 10
            is_first = True
            x0, y0 = 0.0, 0.0
            for row in csv.reader(f):
                if is_first:
                    is_first = False
                    continue
                str_path_no, str_x, str_y = row
                x, y = float(str_x), float(str_y)
                print("* "+str(x)+", "+str(y))
                for i in range(division):
                    self.path_points.append([(x-x0)*i/division+x0, (y-y0)*i/division+y0])
                    print(str((x-x0)*i/division+x0)+", "+str((y-y0)*i/division+y0))
                x0, y0 = x, y

        with open(self.path +  'traceable_pos.csv', 'w') as fw:
            writer = csv.writer(fw, lineterminator='\n')
            with open(self.path +  'path.csv', 'r') as fr:
                i = 0
                for row in csv.reader(fr):
                    if i >= 1:
                        str_no, str_x, str_y = row
                        x, y = float(str_x), float(str_y)
                        if i == 1:
                            x0, y0 = x, y
                        if i >= 2:
                            dx, dy = x - x0, y - y0
                            distance = math.sqrt(dx**2+dy**2)
                            if distance > 0.5:
                                angle = math.atan2(dy, dx)
                                for offset_y in [-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3]:
                                    px, py = x-offset_y*math.sin(angle), y+offset_y*math.cos(angle)
                                    line = [str(px), str(py), str(angle)]
                                    writer.writerow(line)
                                    x0, y0 = x, y
                    i += 1

if __name__ == '__main__':
    ctp = calc_traceable_pos()
    ctp.calc_traceable_pos()

