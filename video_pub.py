"""
Author: Daniel DeBord
Last Updated: July 17 2020

Description:
This code simply publishes a ROS node consisting of a webcam feed
The video is published as a series of rgb8 images under the name 'Videocam'
It was made for testing ros_object_detection.py
"""

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

"""Added Imports"""
import roslib
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import rospy
from std_msgs.msg import String

"""Publishes the video to the node. 
The variable vidcap can be changed to allow for publication of a stored video."""
def main():
	bridge_object = CvBridge()
	pub = rospy.Publisher('Videocam', Image, queue_size=1)
	rospy.init_node('node_name')

	# file input
	#vidcap = cv.VideoCapture(file_name)

	vidcap = cv.VideoCapture(0)

	r = rospy.Rate(24) 
	success, image = vidcap.read()
	while(success and not rospy.is_shutdown()):
		raw = bridge_object.cv2_to_imgmsg(image, encoding="rgb8")
		pub.publish(raw)
		success, image = vidcap.read()
		r.sleep()

if __name__ == '__main__':
    main()
