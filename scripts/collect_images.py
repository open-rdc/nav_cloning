#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import tf
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_net import *
from skimage.transform import resize
import csv
import os
import math
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

class collect_images_node:
    def __init__(self):
        rospy.init_node('collect_images_node', anonymous=True)
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_path/'
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.pos_no = 0
        self.save_img_no = -1
        self.dl = deep_learning(n_action = 1)
        self.load_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_selected_training/20210907_01:11:17/model.net'
        self.dl.load(self.load_path)
        self.target_action = 0

    def callback(self, data):
        try:
            if self.save_img_no != self.pos_no:
                self.save_img_no = self.pos_no
                self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                im_resized = cv2.resize(self.cv_image, dsize=(64, 48))
                cv2.imwrite(self.path+str(self.save_img_no)+".jpg", im_resized)
                img = resize(self.cv_image, (48, 64), mode='constant')
                r, g, b = cv2.split(img)
                imgobj = np.asanyarray([r,g,b])
                self.target_action = self.dl.act(imgobj)
                print(self.target_action)
        except CvBridgeError as e:
            print(e)

    def calc_capture_image_pos(self):
        with open(self.path +  'image_pos.csv', 'w') as fw:
            writer = csv.writer(fw, lineterminator='\n')
            i = 0
            with open(self.path +  'path.csv', 'r') as fr:
                for row in csv.reader(fr):
                    path_no, str_x, str_y = row
                    if i >= 1:
                        x, y = float(str_x), float(str_y)
                        if i == 1:
                            x0, y0 = x, y
                        if i >= 2:
                            distance = math.sqrt((x - x0)**2+(y - y0)**2)
                            if distance > 0.5:
                                angle = math.atan2(y - y0, x - x0)
                                for dy in [-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3]:
                                    line = [str(x-dy*math.sin(angle)), str(y+dy*math.cos(angle)), str(angle)]
                                    writer.writerow(line)
                                x0, y0 = x, y
                    i += 1

    def capture_images(self):
        r = rospy.Rate(10)
        rospy.wait_for_service('/gazebo/set_model_state')
        state = ModelState()
        state.model_name = 'mobile_base'
        x0, y0, the0 = -10.78-0.3, -16.78, 0.01
        with open(self.path +  'result.csv', 'w') as fw:
            writer = csv.writer(fw, lineterminator='\n')
            with open(self.path +  'image_pos.csv', 'r') as fr:
                for row in csv.reader(fr):
                    str_x, str_y, str_the = row
                    x, y, the = float(str_x), float(str_y), float(str_the)
                    dx, dy = x - x0, y - y0
                    state.pose.position.x = math.cos(the0)*dx - math.sin(the0)*dy
                    state.pose.position.y = math.sin(the0)*dx + math.cos(the0)*dy
                    quaternion = tf.transformations.quaternion_from_euler(0, 0, the + the0)
                    state.pose.orientation.x = quaternion[0]
                    state.pose.orientation.y = quaternion[1]
                    state.pose.orientation.z = quaternion[2]
                    state.pose.orientation.w = quaternion[3]
                    try:
                        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                        resp = set_state( state )
                    except rospy.ServiceException, e:
                        print("Service call failed: %s" % e)
                    while self.save_img_no != self.pos_no:
                        r.sleep()
                    line = [str(x), str(y), str(the), str(self.target_action)]
                    writer.writerow(line)
                    self.pos_no += 1

if __name__ == '__main__':
    pc = collect_images_node()
    #pc.calc_capture_image_pos()
    pc.capture_images()

#    DURATION = 0.1
#    r = rospy.Rate(1 / DURATION)
#    while not rospy.is_shutdown():
#        pc.loop()
#        r.sleep()
