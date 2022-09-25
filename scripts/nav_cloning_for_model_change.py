#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_net import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import sys
import tf
from nav_msgs.msg import Odometry

class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.mode = rospy.get_param("/nav_cloning_node/mode", "use_dl_output")
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.dl2 = deep_learning(n_action = self.action_num)
        self.dl3 = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        self.count_waypoint_sub = rospy.Subscriber("count_waypoint", Int8, self.callback_count_waypoint)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.count_waypoint = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.learning = True
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_'+str(self.mode)+'/'
        self.save =  roslib.packages.get_pkg_dir('nav_cloning') + '/data/model/' + self.start_time
        self.save_path = self.save +  '/model1.net'
        self.save_path2 = self.save + '/model2.net'
        self.save_path3 = self.save + '/model3.net'
        #self.load_path = '/home/kiyooka/catkin_ws/src/nav_cloning/data/analysis/conventional/model.net'
        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)
        os.makedirs(self.save)
        

        with open('/home/kiyooka/catkin_ws/src/nav_cloning/data/analysis/conventional/training.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)','x(m)','y(m)', 'the(rad)'])
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_tracker(self, data):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y
        rot = data.pose.pose.orientation
        angle = tf.transformations.euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
        self.pos_the = angle[2]

    def callback_path(self, data):
        self.path_pose = data

    def callback_count_waypoint(self, data):
        self.count_waypoint = data.data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position
        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos.x - path.x)**2 + (pos.y - path.y)**2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        if self.vel.linear.x != 0:
            self.is_started = True
        if self.is_started == False:
            return
        img = resize(self.cv_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img)
        imgobj = np.asanyarray([r,g,b])

        img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img_left)
        imgobj_left = np.asanyarray([r,g,b])

        img_right = resize(self.cv_right_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img_right)
        imgobj_right = np.asanyarray([r,g,b])

        ros_time = str(rospy.Time.now())
        if 8 <= self.count_waypoint <= 15:
            self.count_waypoint = self.count_waypoint - 8
        if 16 <= self.count_waypoint <= 23:
            self.count_waypoint = self.count_waypoint - 16

        if self.episode == 5000:
            self.learning = False
            self.dl.save(self.save_path)
            self.dl2.save(self.save_path)
            self.dl3.save(self.save_path)
            #self.dl.load(self.load_path)
        """
        if self.episode == 10000:
            os.system('killall roslaunch')
            sys.exit()
        """
        if self.learning:
            target_action = self.action
            distance = self.min_distance

            if self.mode == "manual":
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = 0
                action, loss = self.dl.act_and_trains(imgobj, target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
                angle_error = abs(action - target_action)

            elif self.mode == "zigzag":
                action, loss = self.dl.act_and_trains(imgobj, target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
                angle_error = abs(action - target_action)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = 0

            elif self.mode == "use_dl_output":
                if 0 <=self. count_waypoint <= 2:
                    print("model1")
                    action, loss = self.dl.act_and_trains(imgobj, target_action)
                    if abs(target_action) < 0.1:
                        action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
                        action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)

                if 3 <= self.count_waypoint <= 6:
                    print("model2")
                    action, loss = self.dl2.act_and_trains(imgobj, target_action)
                    if abs(target_action) < 0.1:
                        action_left,  loss_left  = self.dl2.act_and_trains(imgobj_left, target_action - 0.2)
                        action_right, loss_right = self.dl2.act_and_trains(imgobj_right, target_action + 0.2)

                if 7 <= self.count_waypoint <= 9:
                    print("model3")
                    action, loss = self.dl3.act_and_trains(imgobj, target_action)
                    if abs(target_action) < 0.1:
                        action_left,  loss_left  = self.dl3.act_and_trains(imgobj_left, target_action - 0.2)
                        action_right, loss_right = self.dl3.act_and_trains(imgobj_right, target_action + 0.2)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action

            elif self.mode == "follow_line":
                action, loss = self.dl.act_and_trains(imgobj, target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
                angle_error = abs(action - target_action)

            # end mode

            print(" episode: " + str(self.episode) + ", loss: " + str(loss) + ", distance: " + str(distance))
            self.episode += 1
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        else:
            if 0 <= self.count_waypoint <= 2:
                print("model1")
                target_action = self.dl.act(imgobj)
            if 3 <= self.count_waypoint <= 6:
                print("model2")
                target_action = self.dl2.act(imgobj)
            if 7 <= self.count_waypoint <= 9:
                print("model3")
                target_action = self.dl3.act(imgobj)
            distance = self.min_distance
            print("TEST MODE: " + " angular:" + str(target_action) + ", distance: " + str(distance))

            self.episode += 1
            angle_error = abs(self.action - target_action)
            line = [str(self.episode), "test", "0", str(angle_error), str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        temp = copy.deepcopy(img_left)
        cv2.imshow("Resized Left Image", temp)
        temp = copy.deepcopy(img_right)
        cv2.imshow("Resized Right Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
