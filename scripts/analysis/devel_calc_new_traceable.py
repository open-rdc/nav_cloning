#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
import math
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
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
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
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/robot_move/'
        #self.load_path = '/home/kiyooka/catkin_ws/src/nav_cloning/data/analysis/use_dl_output/model.net'
        self.load_path = '/home/kiyooka/catkin_ws/src/nav_cloning/data/analysis/follow_path/model.net'
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.reset_count = 1
        self.start_distance = 0.3
        self.start_distance_for_calc = 0.3
        self.offset_ang = 0
        self.angle_reset = 1
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)
        self.gazebo_pos_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_gazebo_pos, queue_size = 2) 
        self.gazebo_pos_x = 0.0
        self.gazebo_pos_y = 0.0
        self.traceable_num_1 = 0
        self.traceable_num_2 = 0
        self.traceable_num_3 = 0
        self.traceable_num_4 = 0
        self.traceable_num_5 = 0
        self.traceable_num_6 = 0
        self.traceable_num = 0
        self.dl.load(self.load_path)
        self.path_points=[]
        self.is_first = True
        self.is_first_evaluation = True
        self.is_finish = False
        with open(self.path + 'path.csv', 'r') as f:
            is_first = True
            for row in csv.reader(f):
                if is_first:
                    is_first = False
                    continue
                str_path_no, str_x, str_y = row
                x, y = float(str_x), float(str_y)
                self.path_points.append([x,y])

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

    def callback_gazebo_pos(self, data):
        self.gazebo_pos_x = data.pose[2].position.x
        self.gazebo_pos_y = data.pose[2].position.y

    def check_distance(self):
        distance_list = []
        for pose in self.path_points:
            path_x = pose[0]
            path_y = pose[1]
            distance = np.sqrt(abs((self.gazebo_pos_x - path_x)**2 + (self.gazebo_pos_y - path_y)**2))
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

    def robot_move(self, x, y, angle): #reset
        r = rospy.Rate(10)
        rospy.wait_for_service('/gazebo/set_model_state')
        state = ModelState()
        state.model_name = 'mobile_base'
        state.pose.position.x = x
        state.pose.position.y = y
        quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state )
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        r.sleep()
        r.sleep()
        r.sleep() #need adjust

    def calc_traceable(self, limit_episode): #evaluation
        if limit_episode:
            return

        else:
            if (self.reset_count-1) % 7 == 1:
                if self.traceable_num == 3:
                    self.traceable_num_1 += 1
            elif (self.reset_count-1) % 7 == 2:
                if self.traceable_num == 3:
                    self.traceable_num_2 += 1
            elif (self.reset_count-1) % 7 == 3:
                if self.traceable_num == 3:
                    self.traceable_num_3 += 1
            elif (self.reset_count-1) % 7 == 5:
                if self.traceable_num == 3:
                    self.traceable_num_4 += 1
            elif (self.reset_count-1) % 7 == 6:
                if self.traceable_num == 3:
                    self.traceable_num_5 += 1
            elif (self.reset_count-1) % 7 == 0:
                if self.traceable_num == 3:
                    self.traceable_num_6 += 1
        
            return self.traceable_num_1,self.traceable_num_2, self.traceable_num_3,self.traceable_num_4,self.traceable_num_5,self.traceable_num_6

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
        number = 0
        self.check_distance()

        #set limit_episode
        if self.episode == 150:
            limit_episode = True
        else:
            limit_episode = False

        if self.reset_count == 8:
            self.is_finish = True

        reset_turn = False

        if self.start_distance > self.min_distance:
            traceable = True
        else:
            traceable = False
            
        print("episode: " + str(self.episode))

        ###########################################################################
        #robot_reset

        if self.is_first:
            with open(self.path + '/traceable_pos.csv', 'r') as f:
                for row in csv.reader(f):
                    number += 1
                    count, str_x, str_y, str_angle, traceable = row
                    if number == self.reset_count:
                        the = float(str_angle) + math.radians(10)
                        the = the - 2.0 * math.pi if the >  math.pi else the
                        the = the + 2.0 * math.pi if the < -math.pi else the
                        self.robot_move(float(str_x),float(str_y),float(the))
                        print("robot_move_first" * 5)
                        print("start_distance: " + str(self.start_distance))
            self.is_first = False


        if self.is_first == False and self.is_finish == False:
            # position is center
            if self.reset_count % 7 == 4:
                self.traceable_num_1,self.traceable_num_2, self.traceable_num_3,self.traceable_num_4,self.traceable_num_5,self.traceable_num_6 = self.calc_traceable(limit_episode)
                self.traceable_num = 0
                self.reset_count += 1
                return

            if traceable or limit_episode:
                # angle 
                if self.angle_reset == 0:
                   self.offset_ang = 10
                elif self.angle_reset == 1:
                   self.offset_ang = 0
                elif self.angle_reset == 2:
                   self.offset_ang = -10

                with open(self.path + '/traceable_pos.csv', 'r') as f:
                    for row in csv.reader(f):
                        number += 1
                        if number == self.reset_count:
                            count, str_x, str_y, str_angle, traceable = row
                            the = float(str_angle) + math.radians(self.offset_ang)
                            the = the - 2.0 * math.pi if the >  math.pi else the
                            the = the + 2.0 * math.pi if the < -math.pi else the
                            self.robot_move(float(str_x),float(str_y),float(the))
                            reset_turn = True
                            print("robot_move")
                            print("start_distance: " + str(self.start_distance))

                if self.angle_reset == 2:
                    if self.reset_count % 7 == 1:
                        self.start_distance = 0.2
                    elif self.reset_count % 7 == 2:
                        self.start_distance = 0.1
                    elif self.reset_count % 7 == 3:
                        self.start_distance = 0.1
                    elif self.reset_count % 7 == 5:
                        self.start_distance = 0.2
                    elif self.reset_count % 7 == 6:
                        self.start_distance = 0.3
                    elif self.reset_count % 7 == 0:
                        self.start_distance = 0.3

                    self.reset_count += 1
                    self.angle_reset = 0

                else:
                    self.angle_reset += 1

                self.episode = 0
                print("reset_count: " + str(self.reset_count))
        ###################################################################################
        #calc

        
            if self.reset_count % 7 == 1:
                self.start_distance_for_calc = 0.3
            elif self.reset_count % 7 == 2:
                self.start_distance_for_calc = 0.3
            elif self.reset_count % 7 == 3:
                self.start_distance_for_calc = 0.2
            elif self.reset_count % 7 == 5:
                self.start_distance_for_calc = 0.1
            elif self.reset_count % 7 == 6:
                self.start_distance_for_calc = 0.1
            elif self.reset_count % 7 == 0:
                self.start_distance_for_calc = 0.2

            # if self.start_distance_for_calc > self.min_distance:
            #     if limit_episode == False:
            #         self.traceable_num += 1

            if reset_turn:
                if self.start_distance_for_calc > self.min_distance:
                    if limit_episode == False:
                        self.traceable_num += 1
            if self.angle_reset == 2:
                self.traceable_num_1,self.traceable_num_2, self.traceable_num_3,self.traceable_num_4,self.traceable_num_5,self.traceable_num_6 = self.calc_traceable(limit_episode)
                print("number: ", str(self.traceable_num))
                print(self.traceable_num_1)
                print(self.traceable_num_2)
                print(self.traceable_num_3)
                print(self.traceable_num_4)
                print(self.traceable_num_5)
                print(self.traceable_num_6)
                self.traceable_num = 0



        if self.is_finish:
            print("fin")
            if traceable or limit_episode == False:
                if limit_episode == False and self.traceable_num == 2:
                    self.traceable_num_6 += 1
                print(self.traceable_num_1/903)
                print(self.traceable_num_2/903)
                print(self.traceable_num_3/903)
                print(self.traceable_num_4/903)
                print(self.traceable_num_5/903)
                print(self.traceable_num_6/903)
                os.system('killall roslaunch')
                sys.exit()

        target_action = self.dl.act(imgobj)
        self.is_first = False
        if self.reset_count != 8:
            self.episode += 1
        line = [str(self.episode), str(self.pos_x), str(self.pos_y), str(self.pos_the)]
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
