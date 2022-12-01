#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../pytorch'))
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
import time
import copy
import tf
from nav_msgs.msg import Odometry

class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.mode = rospy.get_param("/nav_cloning_node/mode", "change_dataset_balance")
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.min_distance = 0.0
        self.episode = 0
        self.vel = Twist()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
        self.load_path = '/home/kiyooka/Downloads/20221103_23_51_10/model_gpu.pt' #specify model
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.position_reset_count = 1
        self.offset_ang = 0
        self.angle_reset_count = 0
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.mode + "/" +   self.start_time)
        self.gazebo_pos_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_gazebo_pos, queue_size = 2) 
        self.gazebo_pos_x = 0.0
        self.gazebo_pos_y = 0.0
        self.traceable_score_1 = 0
        self.traceable_score_2 = 0
        self.traceable_score_3 = 0
        self.traceable_score_4 = 0
        self.traceable_score_5 = 0
        self.traceable_score_6 = 0
        self.move_count = 1
        self.dl.load(self.load_path)
        self.path_points=[]
        self.is_first = True
        self.is_finish = False
        self.start = False
        self.score_list = []
        self.score_list_sum = []
        self.collision_list = [[],[]]
        self.position_change_flag = False
        with open(self.path + self.mode + "/" + 'path.csv', 'r') as f:
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

        # if distance_list:
        self.min_distance = min(distance_list)
        print("ditance: " + str(self.min_distance))


    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.start = data.data
        resp.message = "Training: " + str(self.start)
        resp.success = True
        return resp

    def collision(self):
        collision_flag = False
        self.collision_list[0].append(self.gazebo_pos_x)
        self.collision_list[1].append(self.gazebo_pos_y)
        if len(self.collision_list[0]) == 10:
            average_x = sum(self.collision_list[0]) / len(self.collision_list[0])
            average_y = sum(self.collision_list[1]) / len(self.collision_list[1])
            distance = np.sqrt(abs((self.gazebo_pos_x - average_x)**2 + (self.gazebo_pos_y - average_y)**2))
            self.collision_list[0] = self.collision_list[0][1:]
            self.collision_list[1] = self.collision_list[1][1:]

            if distance < 0.1:
                collision_flag = True
                print("collision")

        return collision_flag


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
        # r.sleep() #need adjust
        # r.sleep() #need adjust
        # r.sleep() #need adjust

    def first_move(self):
        flag = True
        with open(self.path + self.mode + "/" + 'traceable_pos.csv', 'r') as f:
            for row in csv.reader(f):
                if flag:
                    str_x, str_y, str_angle, t = row
                    the = float(str_angle) + math.radians(10)
                    the = the - 2.0 * math.pi if the >  math.pi else the
                    the = the + 2.0 * math.pi if the < -math.pi else the
                    self.robot_move(float(str_x),float(str_y),float(the))
                    print("robot_move_first")
                    flag = False

    def calc_move_pos(self):
        # angle 
        if self.angle_reset_count == 0:
           self.offset_ang = -20.0
        elif self.angle_reset_count == 1:
           self.offset_ang = -10.0
        elif self.angle_reset_count == 2:
           self.offset_ang = 0
        elif self.angle_reset_count == 3:
           self.offset_ang = 10.0
        elif self.angle_reset_count == 4:
           self.offset_ang = 20.0
        # position
        number = 0
        with open(self.path + self.mode + "/" +  '/traceable_pos.csv', 'r') as f:
            for row in csv.reader(f):
                number += 1
                if number == self.position_reset_count:
                    # count, str_x, str_y, str_angle, t = row
                    str_x, str_y, str_angle, t = row
                    the = float(str_angle) + math.radians(self.offset_ang)
                    the = the - 2.0 * math.pi if the >  math.pi else the
                    the = the + 2.0 * math.pi if the < -math.pi else the
                else:
                    pass

        return float(str_x), float(str_y), float(the)

    def check_traceable(self):
        if self.min_distance <= 0.05:
            traceable = True
        else:
            traceable = False
        return traceable

    def eval(self, traceable, angle_num):
        position = (self.position_reset_count - 1) % 7
        if traceable:
            angle_score = 1
        else:
            angle_score = 0
        self.score_list.append(angle_score)
        print("angle_score: " + str(angle_score))
        if self.position_change_flag: # when the x,y position change
            self.score_list_sum.append(self.score_list)
            print("position_score: " + str(self.score_list))
            position_score = sum(self.score_list)/5
            print("position_score: " + str(position_score))
            if position_score == 1:
                if position == 1:
                    self.traceable_score_1 += 1
                elif position == 2:
                    self.traceable_score_2 += 1
                elif position == 3:
                    self.traceable_score_3 += 1
                elif position == 5:
                    self.traceable_score_4 += 1
                elif position == 6:
                    self.traceable_score_5 += 1
                elif position == 0:
                    self.traceable_score_6 += 1

            print("---traceable---")
            print(self.traceable_score_1)
            print(self.traceable_score_2)
            print(self.traceable_score_3)
            print(self.traceable_score_4)
            print(self.traceable_score_5)
            print(self.traceable_score_6)
            print("---traceable---")
            #---------- csv write -----------------
            line = [str(self.score_list), str(position_score)]
            with open(self.path + self.mode + "/" + self.start_time + '/' + 'score.csv', 'a') as fl:
                writer = csv.writer(fl, lineterminator='\n')
                writer.writerow(line)
            # score_list and position_score

            self.score_list = []
            position_score = 0

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        if self.start == False:
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
        collision_flag = False
        self.check_distance() 
        traceable = self.check_traceable() #True or False
        if self.episode > 5:
            collision_flag = self.collision()

        if self.position_reset_count >= 1156:
            self.is_finish = True

        if self.is_first:
            self.first_move()
            self.is_first = False
            return

        if self.is_finish:
            if traceable or collision_flag:
                self.eval(traceable,self.angle_reset_count)
                print("fin")
                print(self.traceable_score_1)
                print(self.traceable_score_2)
                print(self.traceable_score_3)
                print(self.traceable_score_4)
                print(self.traceable_score_5)
                print(self.traceable_score_6)

                line = [str(self.traceable_score_1/165), str(self.traceable_score_2/165), str(self.traceable_score_3/165), str(self.traceable_score_4/165),str(self.traceable_score_5/165),str(self.traceable_score_6/165)]
                with open(self.path + self.mode + "/" + self.start_time + '/' + 'traceable.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(line)
                os.system('killall roslaunch')
                sys.exit()

        else:
            if traceable or collision_flag:
                self.collision_list = [[],[]]
                self.eval(traceable,self.angle_reset_count)
                self.position_change_flag = False
                self.angle_reset_count += 1
                print("angle_reset_count:" + str(self.angle_reset_count))
                print("position_reset_count:" + str(self.position_reset_count))

                if self.position_reset_count %7 == 4: # center position is pass
                    self.position_reset_count += 1
                    self.angle_reset_count = 0
                    print("center_position")

                self.episode = 0
                x,y,the = self.calc_move_pos()
                self.robot_move(x,y,the)
                self.move_count += 1


                # ------------------ num of reset -------------------
                if self.angle_reset_count == 4:
                    self.angle_reset_count = -1
                    self.position_reset_count += 1
                    self.position_change_flag = True
                #------------------------------------------------------




        #
        target_action = self.dl.act(imgobj)
        print("episode:" +str(self.episode))
        print("move_count:" +str(self.move_count))
        self.episode += 1


        if self.episode <= 5:
            self.vel.linear.x = 0.0
            self.vel.angular.z = 0.0
        else:
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
        line_trajectory = [str(self.episode), str(self.gazebo_pos_x), str(self.gazebo_pos_y), str(self.move_count), str(collision_flag)]
        with open(self.path + self.mode + "/" +  self.start_time + '/' + 'trajectory.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(line_trajectory)

        self.nav_pub.publish(self.vel)
        print("------------------"*5)

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
