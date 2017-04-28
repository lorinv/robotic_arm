import sys
sys.path.insert(0, 'computer_vision')
sys.path.insert(0, 'for_arm')
import numpy as np
from predict_labels import TensorBoxPrediction

from selective_search_hog import BlockDetector
import cv2

class ArmState:

    def __init__(self):
        self.detector = TensorBoxPrediction()#BlockDetector()
        self.num_states = 26
        self.screen_sections = 5
        self.cap = cv2.VideoCapture(1)     
        

    def get_num_states(self):
        return self.num_states

    def draw_grid_lines(self,image):        
        for i in range(0, image.shape[1], image.shape[1]/self.screen_sections):
            image[:,i:i+5] = 0
        for i in range(0, image.shape[0], image.shape[0]/self.screen_sections):
            image[i:i+5,:] = 0
        cv2.imshow("State", image)
        cv2.waitKey(25)

        return image

    def get_state(self):   
        for i in range(20):
            ret, image = self.cap.read()
        if image is None:
            print "Failing to get image."
            self.get_state()        
        #center, _, image = self.detector.detect(image)
        center, image = self.detector.detect(image)
        print "Center: %s" % str(center)
        print "Image Size: %s" % str(image.shape)
        image = self.draw_grid_lines(image)
        if center[0] > image.shape[0] or center[1] > image.shape[1] or center[0] == -1:
            return 25
        width, height, ch = image.shape
        per_row = height / self.screen_sections
        per_col = width / self.screen_sections
        x_cell = center[0] / per_row
        y_cell = center[1] / per_col        

        print self.screen_sections * int(y_cell) + int(x_cell)
        return self.screen_sections * int(y_cell) + int(x_cell)

    def display_state_image(self, image):
        cv2.imshow("State", image)
        cv2.waitKey(25)  



