import cv2
import numpy
import math
from enum import Enum

class GripPipeline:
    """
    An OpenCV pipeline generated by GRIP.
    """
    
    def __init__(self):
        """initializes all values to presets or None if need to be set
        """

        self.__hsv_threshold_hue = [29.136690647482013, 114.29541595925298]
        self.__hsv_threshold_saturation = [107.77877697841727, 255.0]
        self.__hsv_threshold_value = [64.20863309352518, 255.0]

        self.hsv_threshold_output = None

        self.__cv_bitwise_not_src1 = self.hsv_threshold_output

        self.cv_bitwise_not_output = None


        self.__mask_mask = self.cv_bitwise_not_output

        self.mask_output = None


    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step HSV_Threshold0:
        self.__hsv_threshold_input = source0
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step CV_bitwise_not0:
        self.__cv_bitwise_not_src1 = self.hsv_threshold_output
        (self.cv_bitwise_not_output) = self.__cv_bitwise_not(self.__cv_bitwise_not_src1)

        # Step Mask0:
        self.__mask_input = source0
        self.__mask_mask = self.cv_bitwise_not_output
        (self.mask_output) = self.__mask(self.__mask_input, self.__mask_mask)


    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __cv_bitwise_not(src1):
        """Computes the per element inverse of an image.
        Args:
            src1: A numpy.ndarray.
        Returns:
            The inverse of the numpy.ndarray.
        """
        return cv2.bitwise_not(src1)

    @staticmethod
    def __mask(input, mask):
        """Filter out an area of an image using a binary mask.
        Args:
            input: A three channel numpy.ndarray.
            mask: A black and white numpy.ndarray.
        Returns:
            A three channel numpy.ndarray.
        """
        return cv2.bitwise_and(input, input, mask=mask)



