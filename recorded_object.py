"""
Author: Daniel DeBord
Last Updated: July 17 2020

Description: 

This code defines the detected_object class. 
The detected object class primarily serves as a mechanism to store information about detected
objects. It stores an recognized objects location, size, and class. 
"""
import numpy as np
class recorded_object(object):
        """Constructor class detected_object(object)
        object_class is class type"""

        # (x,y) is the top left corner of the detected object
        def __init__(self, object_class, x, y, width, height):
            self.object_class = object_class
            self.x = x 
            self.y = y
            self.width = width
            self.height = height

        def get_class(self):
            return self.object_class

        def get_position(self):
            return [self.x, self.y]

        def set_x(self, x):
            self.x = x
        
        def set_y(self, y):
            self.y = y

        def get_height(self):
            return self.height

        def set_height(self, height):
            self.height = height

        def get_width(self):
            return self.width

        def set_width(self, width):
            self.width = width

        def get_characteristics(self):
            return [self.x, self.y, self.width, self.height]

        def get_center(self):
            return [int(self.x + self.width / 2.0), int(self.y + self.height / 2.0)]

        def set_characteristics(self, x, y, width, height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height

        """
        Returns 1 if the point is within the bounds of the detected_object
        0 otherwise
        """
        def is_within(self, targetX, targetY):
            bottomBound = self.y + self.height
            rightBound = self.x + self.width

            if((targetX >= self.x) and (targetX <= rightBound)):
                if((targetY >= self.y) and (targetY <= bottomBound)):
                    return 1
            return 0