#!/usr/bin/env python

# Scripts to collect the route that the robot travels

from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import csv
import os

class path_collector_node:
    def __init__(self):
        rospy.init_node('path_collector_node', anonymous=True)
        self.path_no = 0
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
        self.vel_angular = 0
        self.vel = Twist()
        os.makedirs(self.path)
        with open(self.path +  'path.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['path_no', 'x(m)','y(m)'])
        self.pose_sub = rospy.Subscriber("/tracker", Odometry, self.callback_pose)

    def callback_pose(self, data):
        with open(self.path + 'path.csv', 'a') as f:
            x = data.pose.pose.position.x
            y = data.pose.pose.position.y
            line = [str(self.path_no), str(x), str(y)]
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(line)
        self.path_no += 1

    def callback_vel(self, data):
        self.vel = data
        self.vel_angular = self.vel.angular.z

    def loop(self):
        self.vel.angular.z = self.vel_angular
        self.nav_pub.publish(self.vel)

if __name__ == '__main__':
    pc = path_collector_node()
    DURATION = 0.1
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        pc.loop()
        r.sleep()
