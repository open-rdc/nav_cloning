#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import csv
import math
from matplotlib import pyplot
from matplotlib.patches import ArrowStyle
import numpy as np
from PIL import Image

def arrow():
    rospy.init_node('arrow_node', anonymous=True)
    path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
    image = Image.open(roslib.packages.get_pkg_dir('nav_cloning')+'/maps/map.png').convert("L")
    arr = np.asarray(image)

    fig = pyplot.figure()
    ax = fig.add_subplot(111)

    ax.imshow(arr, cmap='gray', extent=[-10,50,-10,50])
    arrow_dict = dict(arrowstyle = "->", color = "black")
    with open(path + 'result.csv', 'r') as f:
        for row in csv.reader(f):
            str_x, str_y, str_the, str_ang_vel = row
            x, y, the, ang_vel = float(str_x), float(str_y), float(str_the), float(str_ang_vel)
            x1 = x+0.3*math.cos(the+ang_vel*2)
            y1 = y+0.3*math.sin(the+ang_vel*2)
            ax.annotate('', xy=(x1, y1), xytext=(x, y), arrowprops = arrow_dict)

    ax.set_xlim([-5, 30])
    ax.set_ylim([-5, 15])
    pyplot.show()

if __name__ == '__main__':
    arrow()

