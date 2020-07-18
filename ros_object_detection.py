"""
Author: Daniel DeBord
Last Updated: July 17 2020

Description: 

This code uses a video input from a ROS node and performs object recognition on the camera feed
It was modified to run on ROS publishers with custom weights and configurations. 
It also has added tracking and counting objects detected in frame
"""

import cv2 as cv
import argparse
import sys
import numpy as np
import time

""" ROS Imports """
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

""" From Local Imports """
from recorded_object import recorded_object
from tracker import tracker

# parse through arguments
parser = argparse.ArgumentParser()
# Netork Files
parser.add_argument('--weights', '-w', nargs='?', type=str, default="../ConfigStuff/yolov3.weights", help='Marks path to weights')
parser.add_argument('--name_file', '-n', nargs='?', type=str, default="../ConfigStuff/coco.names", help='Marks path to class names')
parser.add_argument('--config', '-c', nargs='?', type=str, default="../ConfigStuff/yolov3.cfg", help='Marks path to config file')
# Custom Parameters
parser.add_argument('--confidence', '-f', type=float, nargs='?', default=0.25, help='Set confidence threshold')
parser.add_argument('--nms', '-s', default=0.4, type=float, nargs='?', help='Set non-maximum supression threshold')
parser.add_argument('--input_size', '-i', default=416, type=int, nargs='?', help='Set size of input image into neural net')
parser.add_argument('--tracking_box', '-tb', default=0.3, type=float, nargs='?', help='Set size of tracking boxes')
parser.add_argument('--subscribed_node', '-sb', default="/Videocam", type=str, nargs='?', help='Set the node to be subscribed to')
parser.add_argument('--frames_before_erasure', '-fbe', default=30, type=int, nargs='?', help='The number of frames before an object is considered no longer present in the frame')

args = parser.parse_args()
# Initialize the parameters
confThreshold = args.confidence #Confidence threshold
nmsThreshold = args.nms #Non-maximum suppression threshold

inputHeight = args.input_size
inputWidth = inputHeight

# get size of tracking box
tracking_size = args.tracking_box

# Load names of classes, classes should be listed in a column with newline delimiters
classesFile = args.name_file

# get the name of the ROS node to subscribe to
subscribedNode = args.subscribed_node

# set the frames before erasure for the tracker
framesBeforeErasure = args.frames_before_erasure

classes = None
with open(classesFile, 'rt') as file:
    # split based on newline delimiter, strip any extra characters
    classes = file.read().rstrip('\n').split('\n')

modelConfiguration = args.config
modelWeights = args.weights

# create network according to whatever demands your system may have
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

"""Object Recog Created to make Object Recog show up on screen"""
class ObjectRecog(object):

    def __init__(self):
        self.bridge_object = CvBridge()

        # Change the subscriber to get a different feed
        self.image_sub = rospy.Subscriber("/Videocam", Image, self.camera_callback)

        # characteristics added for counting
        self.frame_count = 0
        self.start = time.time()
        self.Tracker = tracker(framesBeforeErasure)

    """ Method is called when a new frame is published to the subscribed ros node
    @data is the frame of an image received in OpenCV rgb8 format"""   
    def camera_callback(self, data):
        try:
            # We select rgb8 because its the OpneCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="rgb8")
        except CvBridgeError as e:
            print(e)

        # get image from node
        window = 'Object Recognition'
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        self.frame = cv_image

        # Create a tensor from the frame (decimals included to be python 2 friendly)
        tensorBlob = cv.dnn.blobFromImage(self.frame, 1.0/255.0, (inputWidth, inputHeight), [0,0,0], 0, crop=False)

        # input the tensor blob to the network
        net.setInput(tensorBlob)
        # get outputs from a forward pass through the network (only one in YOLO)
        # use get output names to get the outputs that matter
        netOutputs = net.forward(self.getOutputsNames())

        # performan non-maximum supression and remove low confidence guesses
        self.nonmaxSurpression(netOutputs)


        # gather efficiency information                
        current = time.time() #get current time
        self.frame_count = self.frame_count + 1
        print("FPS: %f" % (self.frame_count/(current-self.start)))

        print("Total Number Seen: %f" % self.Tracker.get_count())

        #update tracking list data
        self.Tracker.update_frames()
        self.Tracker.clean_lists()
        
        cv.imshow(window, self.frame)
        cv.waitKey(1)


    """ Get the names of the output layers """
    def getOutputsNames(self):
        layers = net.getLayerNames()
        # get a list of layers just before the unconnected output layers
        return [layers[index[0] - 1] for index in net.getUnconnectedOutLayers()]

    """ Draw the bounding box of any objects detected in the frame 
    @classId is the index of the class detected in numerical form
    @conf is the confidence threshold
    @(x, y) is the top right corner of the rectangle
    @width and height are the width and heith of the image"""
    def drawPrediction(self, classID, confidence, x, y, width, height):
        # add bounding box
        # convert into vertices of box
        left = x
        top = y
        right = left + width
        bottom = top + height

        cv.rectangle(self.frame, (left, top), (right, bottom), (255, 255, 0), 3)
        label = '%.2f' % confidence
        label = '%s: %s' % (classes[classID], label) 

        textSize, bottomLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        labelTop = max(textSize[1], top)


        # draw a smaller box representing the tracking bounds
        trackWidth =  int(width*2*tracking_size)
        trackHeight = int(height*2*tracking_size)
        trackX = x + int((0.5 - tracking_size) * width)
        trackY = y + int((0.5 - tracking_size) * height)

        # verify if this object has been recorded in the past
        trackColor = (0, 0, 0)
        currentObject = recorded_object(classID, trackX, trackY, trackWidth, trackHeight)
        index = self.Tracker.check_object(currentObject)
        if(index == -1):
            # new object detected
            self.Tracker.add_object(currentObject)
            trackColor = (255, 255, 255)
        else:
            # update position of old object
            self.Tracker.update_object(currentObject, index)

        cv.rectangle(self.frame, (trackX, trackY), ((trackX+trackWidth), (trackY+trackHeight)), trackColor, 3)


        # add box for the label
        cv.rectangle(self.frame, (left, labelTop - int(round(1.5*textSize[1]))), (left + int(round(1.25*textSize[0])), labelTop + bottomLine), (255, 255, 0), cv.FILLED)
        # add label
        cv.putText(self.frame, label, (left, labelTop),  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))


    """ Removes the bounding boxes with lower confidences and performs non-max surpression
    @frame contains the image frame being manipulated
    @outs outputs of the neural network"""
    def nonmaxSurpression(self, outputs):
        # remove low confidence boxes
        classIDs = []
        confidenceSet = []
        bounds = []
        winHeight = self.frame.shape[0]
        winWidth = self.frame.shape[1]

        # search outputs for high confidence detections
        for out in outputs:
            for detectedObject in out:
                scoreSet = detectedObject[5:]
                classID = np.argmax(scoreSet) # get the most likely type of the detection
                confidence = scoreSet[classID] # get the highest confidence

                # only add detection if it passes likelihood threshold 
                if(confidence > confThreshold):
                    confidenceSet.append(float(confidence))
                    classIDs.append(classID)
                    # get bounds
                    (centerX, centerY, boxWidth, boxHeight) = [int(detectedObject[0]*winWidth), int(detectedObject[1]*winHeight), 
                        int(detectedObject[2]*winWidth), int(detectedObject[3]*winHeight)]
                    left = int(centerX - (boxWidth / 2.0))
                    top = int(centerY - (boxHeight / 2.0))
                    bounds.append([left, top, boxWidth, boxHeight])

        # perform non maxim supression because YOLOv3 does not include it by default
        indices = cv.dnn.NMSBoxes(bounds, confidenceSet, confThreshold, nmsThreshold)

        # draw boxes
        if(len(indices) > 0):
            for i in indices.flatten():
                self.drawPrediction(classIDs[i], confidenceSet[i], bounds[i][0], bounds[i][1], bounds[i][2], bounds[i][3])


"""Begin the process of looking for ROS inputs"""
def main():
    ObjectRecog()
    rospy.init_node('object_id_node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()