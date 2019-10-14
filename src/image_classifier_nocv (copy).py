#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('farmaid_classification')
import sys
import rospy
import cv2
import torch
import os
from PIL import Image as Img
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pytorch_load_test import get_model
import numpy as np

def pil2tensor(image):
    a = np.asarray(image)
    if a.ndim == 2: a = np.expand_dims(a,2)
    a = np.transpose(a, (1,0,2))
    a = np.transpose(a, (2,1,0))
    return torch.from_numpy(a.astype(np.float32, copy=False))

class image_converter(object):
    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=1)
        self.string_pub = rospy.Publisher("string_topic",String,queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/csi_cam_0/image_raw",Image, self.callback)
        self.model = get_model()
        self.load_model()
        self.sigmoid = torch.nn.Sigmoid()
    def load_model(self):
        source = "/home/nvidia/workspace/catkin_ws/src/farmaid_classification/models/farmaid_model_pytorch"
        try:
            self.model.load_state_dict(torch.load(source))
            self.model.eval()
            rospy.loginfo("model loaded")
        except IOError as e:
            print("model not found in "+ str(os.getcwd()))
            rospy.loginfo("model not found in "+ str(os.getcwd()))

    def callback(self, data):
        rospy.loginfo(type(data.data))
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.string_pub.publish(str(e))
            print(e)
        (rows, cols, channels) = cv_image.shape
        
        if self.model is not None:
            img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_img = Img.fromarray(img)
            pil_img.resize((128,128))
            torch_input = pil2tensor(pil_img)
            rospy.loginfo('processing image')
            
            result = self.model(torch_input[None])
            result = self.sigmoid(result)
            rospy.loginfo("got result")
            msg = String()
            msg.data = str(result)
            self.string_pub.publish(msg)
            rospy.loginfo(str(result))
        else:
            rospy.loginfo("model not found or image not loaded")
                
        if cols > 60 and rows > 60:
            cv2.circle(cv_image, (rows//2,cols//2), 100, (0,200,255))
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

def main(args):
    ic = image_converter()
    rospy.init_node("classifier",anonymous=True)
    try:
        rospy.spin()
        #while not rospy.is_shutdown():
        #    continue
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)   
