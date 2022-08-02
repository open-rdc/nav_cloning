#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import csv
import math
from matplotlib import pyplot
from matplotlib.patches import Circle, Polygon
import numpy as np
from PIL import Image


def draw_traceable_pos():
    rospy.init_node('draw_traceable_pos_node', anonymous=True)
    path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/robot_move/'
    image = Image.open(roslib.packages.get_pkg_dir('nav_cloning')+'/maps/map.png').convert("L")
    arr = np.asarray(image)
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(arr, cmap='gray', extent=[-10,50,-10,50])
    vel = 0.2
    arrow_dict = dict(arrowstyle = "->", color = "black")
    count = 0

    with open(path + 'path.csv', 'r') as f:
        is_first = True
        points=[]
        for row in csv.reader(f):
            if is_first:
                is_first = False
                continue
            str_path_no, str_x, str_y = row
            x, y = float(str_x), float(str_y)
            points.append([x,y])
            patch = Polygon(xy=points, closed=False, fill=False, linewidth=1.5, edgecolor="gray")
        ax.add_patch(patch)

    # with open(path + 'use_dl_output/test/traceable_pos.csv', 'r') as f:
    # with open(path + 'follow_line/test/traceable_pos.csv', 'r') as f:
    # with open(path + '300_use_dl_output/test/traceable_pos.csv', 'r') as f:
    with open(path + '300_follow_line/test/traceable_pos.csv', 'r') as f:
    # with open(path + 'gomi/test/traceable_pos.csv', 'r') as f:
        for row in csv.reader(f):
            str_x, str_y, str_traceable = row
            x, y, traceable = float(str_x), float(str_y), float(str_traceable)
            if traceable == 1.0:
                patch = Circle(xy=(x, y), radius=0.05, facecolor="green")
                count += 1
            # elif traceable > 0.6:
            #     patch = Circle(xy=(x, y), radius=0.05, facecolor="orange")
            # elif traceable > 0.3:
            #     patch = Circle(xy=(x, y), radius=0.05, facecolor="blue")
            else:
                patch = Circle(xy=(x, y), radius=0.05, facecolor="red")
                #patch = Circle(xy=(x, y), radius=(0.2+traceable*0.8)*0.04, facecolor="red")
            ax.add_patch(patch)
            
    ax.set_xlim([-5, 30])
    ax.set_ylim([-5, 15])
    pyplot.show()
    print(count)

if __name__ == '__main__':
    draw_traceable_pos()

