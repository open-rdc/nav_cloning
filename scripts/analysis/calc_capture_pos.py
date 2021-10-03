#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import csv
import math

class calc_capture_pos:
    def __init__(self):
        rospy.init_node('calc_capture_pos_node', anonymous=True)
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'

    def calc_pos(self):
        offset_angle = 0
        with open(self.path +  'capture_pos.csv', 'w') as fw:
            writer = csv.writer(fw, lineterminator='\n')
            i = 0
            with open(self.path +  'path.csv', 'r') as fr:
                for row in csv.reader(fr):
                    if i >= 1:
                        path_no, str_x, str_y = row
                        x, y = float(str_x), float(str_y)
                        if i == 1:
                            x0, y0 = x, y
                        if i >= 2:
                            distance = math.sqrt((x - x0)**2+(y - y0)**2)
                            if distance > 0.5:
                                angle = math.atan2(y - y0, x - x0)
                                direction = angle + math.pi / 180 * offset_angle
                                direction = direction - 2.0 * math.pi if direction >  math.pi else direction
                                direction = direction + 2.0 * math.pi if direction < -math.pi else direction
                                for dy in [-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3]:
                                    line = [str(x-dy*math.sin(angle)), str(y+dy*math.cos(angle)), str(direction)]
                                    writer.writerow(line)
                                x0, y0 = x, y
                    i += 1

if __name__ == '__main__':
    ccp = calc_capture_pos()
    ccp.calc_pos()

