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
    image = Image.open(roslib.packages.get_pkg_dir('nav_cloning')+'/maps/map.png').convert("L")
    arr = np.asarray(image)
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(arr, cmap='gray', extent=[-10,50,-10,50])
    vel = 0.2
    arrow_dict = dict(arrowstyle = "->", color = "black")
    count = 0
    # with open('/home/kiyooka/catkin_ws/src/nav_cloning/data/result_use_dl_output/rate_change_10/training.csv', 'r') as f:
    # with open('/home/kiyooka/catkin_ws/src/nav_cloning/data/result_use_dl_output/rate_change_2000step_1/training.csv', 'r') as f:
    with open('/home/kiyooka/catkin_ws/src/nav_cloning/data/result_with_dir_use_dl_output/5000step31/training.csv', 'r') as f:
    # with open('/home/kiyooka/catkin_ws/src/nav_cloning/data/result_change_dataset_balance/add_10/training.csv', 'r') as f:
    # with open('/home/kiyooka/catkin_ws/src/nav_cloning/data/result_selected_training/20220817_21:20:01/training.csv', 'r') as f:
    # with open('/home/kiyooka/catkin_ws/src/nav_cloning/data/result_rate_change/5000step40/training.csv', 'r') as f:
        for row in csv.reader(f):
                # str_step, mode,loss, angle_error,distance,str_x, str_y, str_the = row
                str_step, mode,distance,str_x, str_y, str_the = row
                # str_step, mode,distance,str_x, str_y = row
                # str_step, mode,distance,str_x, str_y, str_the,learning_count = row
                # step, x, y = int(str_step), float(str_x), float(str_y)
                # patch = Circle(xy=(x, y), radius=0.08, facecolor="gray") 
                # ax.add_patch(patch)
                # else:
                if mode == "test":
                    x, y = float(str_x), float(str_y)
                    patch = Circle(xy=(x, y), radius=0.08, facecolor="gray") 
                    ax.add_patch(patch)
        else:
                    pass
        # else:
        #     pass

    #
    # with open(path + 'path.csv', 'r') as f:
    #     is_first = True
    #     points=[]
    #     for row in csv.reader(f):
    #         if is_first:
    #             is_first = False
    #             continue
    #         str_path_no, str_x, str_y = row
    #         x, y = float(str_x), float(str_y)
    #         points.append([x,y])
    #         patch = Polygon(xy=points, closed=False, fill=False, linewidth=0.5, edgecolor="red")
    #     ax.add_patch(patch)
            
    ax.set_xlim([-5, 30])
    ax.set_ylim([-5, 15])
    pyplot.show()

if __name__ == '__main__':
    draw_training_pos()

