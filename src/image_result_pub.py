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



class image_converter(object):
    def __init__(self):
        self.image_pub = rospy.Publisher("image_result",Image,queue_size=1)
        self.bridge = CvBridge()
        self.string_sub = rospy.Subscriber("/string_topic",Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.string_pub.publish(str(e))
            print(e)
        (rows, cols, channels) = cv_image.shape
        
        if cv_image is not None:
            cv2.putPutText(cv_image, data.data, (rows,cols), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

def main(args):
    ic = image_converter()
    rospy.init_node("classifier",anonymous=True)
    while not rospy.is_shutdown():
        continue
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)   
