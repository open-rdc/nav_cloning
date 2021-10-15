#!/usr/bin/env python
from __future__ import print_function
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
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
import math
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

class calc_ang_vel:
    def __init__(self):
        rospy.init_node('calc_ang_vel_node', anonymous=True)
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.pos_no = 0
        self.save_img_no = 0
        self.dl = deep_learning(n_action = 1)
        self.load_path = self.path + 'model.net'
        self.dl.load(self.load_path)
        self.target_action = 0

    def callback(self, data):
        try:
            if self.save_img_no != self.pos_no:
                self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                im_resized = cv2.resize(self.cv_image, dsize=(64, 48))
                cv2.imwrite(self.path+str(self.save_img_no)+".jpg", im_resized)
                img = resize(self.cv_image, (48, 64), mode='constant')
                r, g, b = cv2.split(img)
                imgobj = np.asanyarray([r,g,b])
                self.target_action = self.dl.act(imgobj)
                print(self.target_action)
                self.save_img_no = self.pos_no
        except CvBridgeError as e:
            print(e)

    def calc_ang_vel(self):
        r = rospy.Rate(10)
        rospy.wait_for_service('/gazebo/set_model_state')
        state = ModelState()
        state.model_name = 'mobile_base'
        x0, y0, the0 = 0.0, 0.0, 0.0
        with open(self.path +  'result.csv', 'w') as fw:
            writer = csv.writer(fw, lineterminator='\n')
            with open(self.path +  'capture_pos.csv', 'r') as fr:
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
                    r.sleep()
                    r.sleep()
                    r.sleep() #need adjust
                    self.pos_no += 1
                    while self.save_img_no != self.pos_no:
                        r.sleep()
                    line = [str(x), str(y), str(the), str(self.target_action)]
                    writer.writerow(line)

if __name__ == '__main__':
    cav = calc_ang_vel()
    cav.calc_ang_vel()

