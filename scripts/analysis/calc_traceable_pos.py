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

class calc_traceable_pos:
    def __init__(self):
        rospy.init_node('calc_traceable_pos_node', anonymous=True)
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.pos_no = 0
        self.save_img_no = 0
        self.dl = deep_learning(n_action = 1)
        self.load_path = self.path + 'model.net'
        self.dl.load(self.load_path)
        self.target_action = 0
        self.path_points=[]
        self.r = rospy.Rate(10)
        rospy.wait_for_service('/gazebo/set_model_state')
        self.state = ModelState()
        self.state.model_name = 'mobile_base'

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

    def check_traceable(self, x, y, angle, dy):
        traceable = 0.0
        vel = 0.2
        for offset_ang in [-5, 0, 5]:
            the = angle + math.radians(offset_ang)
            the = the - 2.0 * math.pi if the >  math.pi else the
            the = the + 2.0 * math.pi if the < -math.pi else the
            self.state.pose.position.x = x
            self.state.pose.position.y = y
            quaternion = tf.transformations.quaternion_from_euler(0, 0, the)
            self.state.pose.orientation.x = quaternion[0]
            self.state.pose.orientation.y = quaternion[1]
            self.state.pose.orientation.z = quaternion[2]
            self.state.pose.orientation.w = quaternion[3]
            try:
                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                resp = set_state( self.state )
            except rospy.ServiceException, e:
                print("Service call failed: %s" % e)
            self.r.sleep()
            self.r.sleep()
            self.r.sleep() #need adjust
            self.pos_no += 1
            while self.save_img_no != self.pos_no:
                self.r.sleep()
            ang_vel = self.target_action
            if ang_vel == 0:
                ang_vel = 0.001
            radius = vel/ang_vel
            cx = x-radius*math.sin(the)
            cy = y+radius*math.cos(the)
            if self.target_action > 0.0:
                epx = cx+radius*math.cos(the-math.pi/2+0.5/radius)
                epy = cy+radius*math.sin(the-math.pi/2+0.5/radius)
            else:
                epx = cx-radius*math.cos(the+math.pi/2+0.5/radius)
                epy = cy-radius*math.sin(the+math.pi/2+0.5/radius)
            min_len_sq = 10000.0
            for p in self.path_points:
                l_sq = (epx - p[0])**2+(epy - p[1])**2
                if l_sq < min_len_sq:
                    min_len_sq = l_sq
            min_len = math.sqrt(min_len_sq)
            if min_len < abs(dy) or min_len < 0.1:
                traceable += 1.0
            print("dy: "+str(dy)+", ang_vel: "+str(ang_vel)+", min_len: "+str(min_len))
        return traceable/3.0

    def calc_traceable_pos(self):
        with open(self.path +  'path.csv', 'r') as f:
            division = 10
            is_first = True
            x0, y0 = 0.0, 0.0
            for row in csv.reader(f):
                if is_first:
                    is_first = False
                    continue
                str_path_no, str_x, str_y = row
                x, y = float(str_x), float(str_y)
                print("* "+str(x)+", "+str(y))
                for i in range(division):
                    self.path_points.append([(x-x0)*i/division+x0, (y-y0)*i/division+y0])
                    print(str((x-x0)*i/division+x0)+", "+str((y-y0)*i/division+y0))
                x0, y0 = x, y

        with open(self.path +  'traceable_pos.csv', 'w') as fw:
            writer = csv.writer(fw, lineterminator='\n')
            with open(self.path +  'path.csv', 'r') as fr:
                i = 0
                for row in csv.reader(fr):
                    if i >= 1:
                        str_no, str_x, str_y = row
                        x, y = float(str_x), float(str_y)
                        if i == 1:
                            x0, y0 = x, y
                        if i >= 2:
                            dx, dy = x - x0, y - y0
                            distance = math.sqrt(dx**2+dy**2)
                            if distance > 0.5:
                                angle = math.atan2(dy, dx)
                                for offset_y in [-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3]:
                                    px, py = x-offset_y*math.sin(angle), y+offset_y*math.cos(angle)
                                    traceable = self.check_traceable(px, py, angle, offset_y)
                                    line = [str(px), str(py), str(angle), str(traceable)]
                                    writer.writerow(line)
                                    x0, y0 = x, y
                    i += 1

if __name__ == '__main__':
    ctp = calc_traceable_pos()
    ctp.calc_traceable_pos()

