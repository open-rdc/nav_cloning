#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import csv
import math
from matplotlib import pyplot
from matplotlib.patches import Arrow, Arc
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
    vel = 0.2
    arrow_dict = dict(arrowstyle = "->", color = "black")
    with open(path + 'result.csv', 'r') as f:
        for row in csv.reader(f):
            str_x, str_y, str_the, str_ang_vel = row
            x, y, the, ang_vel = float(str_x), float(str_y), float(str_the), float(str_ang_vel)
            if ang_vel == 0:
                ang_vel = 0.001
            r = vel/ang_vel
            cx = x-r*math.sin(the)
            cy = y+r*math.cos(the)
            if ang_vel > 0:
                patch = Arc(xy=(cx, cy), width=r*2, height=r*2, angle=math.degrees(the)-90, theta1=0, theta2=math.degrees(0.4/r))
                arx1 = cx+r*math.cos(the-math.pi/2+0.4/r)
                ary1 = cy+r*math.sin(the-math.pi/2+0.4/r)
                arx0 = cx+r*math.cos(the-math.pi/2+0.35/r)
                ary0 = cy+r*math.sin(the-math.pi/2+0.35/r)
                ax.annotate('', xy=(arx1, ary1), xytext=(arx0, ary0), arrowprops = arrow_dict)
            else:
                patch = Arc(xy=(cx, cy), width=-r*2, height=-r*2, angle=math.degrees(the)+90, theta1=math.degrees(0.4/r), theta2=0)
                arx1 = cx-r*math.cos(the+math.pi/2+0.4/r)
                ary1 = cy-r*math.sin(the+math.pi/2+0.4/r)
                arx0 = cx-r*math.cos(the+math.pi/2+0.3/r)
                ary0 = cy-r*math.sin(the+math.pi/2+0.3/r)
                ax.annotate('', xy=(arx1, ary1), xytext=(arx0, ary0), arrowprops = arrow_dict)
            ax.add_patch(patch)
            
    ax.set_xlim([-5, 30])
    ax.set_ylim([-5, 15])
    pyplot.show()

if __name__ == '__main__':
    arrow()

