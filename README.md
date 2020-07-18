ROS YOLOv3 Object Detections

This code perform YOLOv3 based object detection and recognition through a video feed provided via a ROS node. The objects are then counted and tracked as they move through the frame. 
To read more about YOLOv3 check here: https://pjreddie.com/darknet/yolo/. 
YoloV3 config files and weights made for the COCO dataset may also be found through the above link. 

The code is broken down into four files:

1. ros_object_detection_counting.py: This file performs YOLOv3 based object detection on the camera feed subscribed to via ROS.  Use the -h flag to see various configuration options with the system

2. video_pub.py: This file was made for simple, off-the-robot testing of the system. It publishes a webcam feed or a stored video to be handled by the detection script.

3. detected_object.py: This class is basically just a structure created for storing data about an object detected by the main script

4. object_tracker.py: This class tracks previously detected objects and counts how many individual objects have been detected so far. 

Dependencies:
ROS (Tested on Melodic)
numpy (1.16)
OpenCV (Tested on version 4.2)
cv_bridge (Tested on 1.13)
roslib (1.14)


