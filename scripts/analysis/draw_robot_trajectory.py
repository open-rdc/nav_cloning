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


def draw_training_pos():
    rospy.init_node('draw_training_pos_node', anonymous=True)
    path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/robot_move/'
    image = Image.open(roslib.packages.get_pkg_dir('nav_cloning')+'/maps/map.png').convert("L")
    arr = np.asarray(image)
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(arr, cmap='gray', extent=[-10,50,-10,50])
    vel = 0.2
    arrow_dict = dict(arrowstyle = "->", color = "black")
    count = 0
    with open(path + 'use_dl_output/training.csv', 'r') as f:
    #with open(path + 'follow_line/training.csv', 'r') as f:
        for row in csv.reader(f):
            str_step, str_x, str_y, str_the = row
            step, x, y, the = int(str_step), float(str_x), float(str_y), float(str_the)
            if step == 50:
                count+=1
            if step > 0:
              for i in range(869,876): # about 445 = first_turn , 575 , 715, 869
                if count == i:
                    patch = Circle(xy=(x, y), radius=0.01, facecolor="gray")
                    ax.add_patch(patch)


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
            patch = Polygon(xy=points, closed=False, fill=False, linewidth=0.5, edgecolor="red")
        ax.add_patch(patch)
            
    ax.set_xlim([-5, 30])
    ax.set_ylim([-5, 15])
    pyplot.show()

if __name__ == '__main__':
    draw_training_pos()

