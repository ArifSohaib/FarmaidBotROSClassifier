#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('farmaid_classification')
import sys
import rospy
import cv2
import torch
from torchvision import transforms
import os
from PIL import Image as Img
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pytorch_load_test import get_model
import numpy as np

def fix_np(image):
    a = np.asarray(image)
    if a.ndim == 2: a = np.expand_dims(a,2)
    a = np.transpose(a, (1,0,2))
    a = np.transpose(a, (2,1,0))
    return a

class image_converter(object):
    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=1)
        self.string_pub = rospy.Publisher("string_topic",String,queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/csi_cam_0/image_raw",Image, self.callback)
        self.model = get_model()
        self.load_model()
        self.sigmoid = torch.nn.Sigmoid().to(torch.device("cuda"))
    def load_model(self):
        source = "/home/nvidia/workspace/catkin_ws/src/farmaid_classification/models/farmaid_model_pytorch"
        try:
            self.model.load_state_dict(torch.load(source))
            self.model.eval()
            self.model.to(torch.device("cuda"))
            rospy.loginfo("model loaded")
        except IOError as e:
            print("model not found in "+ str(os.getcwd()))
            rospy.loginfo("model not found in "+ str(os.getcwd()))

    def callback(self, data):
        rospy.loginfo(type(data.data))
        try:
            img = np.frombuffer(data.data,dtype=np.uint8)\
                     .reshape(3,data.height,data.width)\
                     .astype(np.uint8)
        except:
            rospy.logerr("couldent convert image")
            return
        if img is not None:
            #NOTE: PIL.Image has to be used as Img as there is a name conflict
            #torch_input = Img.fromarray(img)
            #rospy.loginfo(str(type(torch_input)))
            tfms = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.Resize(128),
                           transforms.ToTensor()])
            torch_input = tfms(img).view(-1,3,128,128).to(torch.device("cuda"))
            result = self.model(torch_input)
            result = self.sigmoid(result)
            rospy.loginfo(str(result))
            self.string_pub.publish(String(str(result)))

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
