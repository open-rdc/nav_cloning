#!/usr/bin/env python3
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
from std_msgs.msg import Float32, Int8
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import random
import math
import sys

class cource_following_learning_node:
	def __init__(self):
		rospy.init_node('cource_following_learning_node', anonymous=True)
		self.action_num = rospy.get_param("/LiDAR_based_learning_node/action_num", 1)
		print("action_num: " + str(self.action_num))
		self.dl = deep_learning(n_action = self.action_num)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera_center/image_raw", Image, self.callback)
		self.image_left_sub = rospy.Subscriber("/camera_left/image_raw", Image, self.callback_left_camera)
		self.image_right_sub = rospy.Subscriber("/camera_right/image_raw", Image, self.callback_right_camera)
		self.vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.callback_vel)
		self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
		self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
		self.action = 0.0
		self.episode = 0
		self.loss = 0.0
		self.angle_error = 0.0
		self.vel = Twist()
		self.cv_image = np.zeros((480,640,3), np.uint8)
		self.cv_left_image = np.zeros((480,640,3), np.uint8)
		self.cv_right_image = np.zeros((480,640,3), np.uint8)
		self.learning = True
		self.select_dl = False
		self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
		self.path = '/home/rdclab/learning_ws/src/nav_cloning/data/'
		self.previous_reset_time = 0
		self.start_time_s = rospy.get_time()
		os.makedirs(self.path + self.start_time)
		self.dl.load()

		with open(self.path + self.start_time + '/' +  'reward.csv', 'w') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)'])

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

	def callback_pose(self, data):
		pos = data.pose.pose.position

	def callback_vel(self, data):
		self.vel = data
		self.action = self.vel.angular.z

	def callback_dl_training(self, data):
		resp = SetBoolResponse()
		self.dl.save()
		resp.message = "save"
#		resp.success = True
		return resp

	def loop(self):
		if self.cv_image.size != 640 * 480 * 3:
			return
		if self.cv_left_image.size != 640 * 480 * 3:
			return
		if self.cv_right_image.size != 640 * 480 * 3:
			return

		if self.vel.linear.x == None:
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

		if self.episode ==10730:
			self.learning = False
			self.dl.save()
#			self.dl.load()

		if self.learning:
			target_action = self.action
			self.action = self.dl.act(imgobj)
			self.angle_error = abs(self.action - target_action)

			if self.angle_error > 0.07:
				self.action, self.loss = self.dl.act_and_trains(imgobj, target_action)
				if abs(target_action) < 0.1:
					action_left,  self.loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
					action_right, self.loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)

			print(" episode: " + str(self.episode) + ", self.loss: " + str(self.loss) + ", angle_error: " + str(self.angle_error))
			self.episode += 1
			line = [str(self.episode), "training", str(self.loss), str(self.angle_error)]
			with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
				writer = csv.writer(f, lineterminator='\n')
				writer.writerow(line)

		temp = copy.deepcopy(img)
		cv2.imshow("Resized Image", temp)
		temp = copy.deepcopy(img_left)
		cv2.imshow("Resized Left Image", temp)
		temp = copy.deepcopy(img_right)
		cv2.imshow("Resized Right Image", temp)
		cv2.waitKey(1)

if __name__ == '__main__':
	rg = cource_following_learning_node()
	DURATION = 0.2
	r = rospy.Rate(1 / DURATION)
	while not rospy.is_shutdown():
		rg.loop()
		r.sleep()
