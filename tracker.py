"""
Author: Daniel DeBord
Last Updated: July 17 2020

Description: 

This code defines the object_tracker class. 
The object_tracker class keeps track of objects seen in the past images based on their location
relative to the frame. If another object is seen near that spot on a frame shortly after this one, it is assumed to the 
same object and not counted as new. Positions can be updated over time as objects move around the frame of the image.

If an object is not recognized after a certain amount of frames, it is removed from the list. 
This allows the robot to count new sea urchins as they appear in the moving video feed of the robot while not counting
sea urchins seen earlier, provided the robot does not explore the same place twice. 
"""


import numpy as np
from recorded_object import recorded_object

class tracker(object):
        """
        Initialize the object_tracker object. Can be initialized with or without a list.
        Frames before clean is how long an image can be out of detection before being eliminated
        """
        def __init__(self, framesBeforeClean = 30, objectList=[]):
            self.objectList = objectList
            self.frameCountList = []
            self.currentCount = 0 # the count of how many objects have been detected
            self.framesBeforeClean = framesBeforeClean

        """ Adds object to list being tracked """
        def add_object(self, new_object):
            self.objectList.append(new_object)
            self.frameCountList.append(self.framesBeforeClean)
            self.currentCount = self.currentCount + 1

        """ Check to see if an object occupies the same space as an already detected object
        Returns the index of the detected object if it does, returns -1 if not object is detected"""
        def check_object(self, target):
            numIndex = 0
            targetCenter = target.get_center()
            for detection in self.objectList:
                if(detection.get_class() == target.get_class() and 
                    detection.is_within(targetCenter[0], targetCenter[1])):
                    return numIndex
                numIndex = numIndex + 1
            return -1

        """ Remove an object at an index received as an input """
        def remove_index(self, index):
            if(index < 0):
                return -1
            del self.frameCountList[index]
            del self.objectList[index]
            return 1

        """ Removes object received as an input from list being tracked
        Objects are determines as being the same if they occupy the same space """
        def remove_object(self, target):
            potentialIndex = self.check_object(target)
            if(potentialIndex == -1):
                return -1
            del self.frameCountList[potentialIndex]
            del self.objectList[potentialIndex]
            return 1

        """ Returns number of objects seen so far """ 
        def get_count(self):
            return self.currentCount

        """ This method updates the characteristics of an object detected as a it moves
        around an image. It also resets the frames until removal for that object """
        def update_object(self, newObject, index):
            self.objectList[index] = newObject
            self.frameCountList[index] = self.framesBeforeClean

        """ Decreases the number of frames until removal """
        def update_frames(self):
            for index in np.arange(0, len(self.frameCountList)):
                self.frameCountList[index] = self.frameCountList[index] - 1

        """ Updates the list and removes cases that have expired """
        def clean_lists(self):
            index = 0
            while(index < len(self.frameCountList)):
                if(self.frameCountList[index] <= 0):
                    # remove from list
                    del self.frameCountList[index]
                    del self.objectList[index]
                    index = index - 1
                index = index + 1

