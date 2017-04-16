import sys
sys.path.insert(0, 'computer_vision')

from selective_search_hog import BlockDetector
import cv2

class ArmState:

    def __init__(self):
        self.detector = BlockDetector()
        self.num_states = 4
        self.screen_sections = self.num_states - 1

    def get_num_states(self):
        return self.num_states

    
    def get_state(self):
        center, _, image = self.detector.detect()
        for i in range(0, image.shape[1], image.shape[1]/self.screen_sections):
            image[:,i:i+5] = 0
        cv2.imshow("State", image)
        cv2.waitKey(25)
        if center[0] < image.shape[1] / self.screen_sections and center[0] > 0:
            return 0
        if center[0] > image.shape[1] / self.screen_sections and center[0] < image.shape[1] / self.screen_sections * 2:
            return 1
        if center[0] > image.shape[1] / self.screen_sections * 2:
            return 2
        else:
            return 3
    
    '''
    def get_state(self):
        center, _, image = self.detector.detect()
        for i in range(0, image.shape[1], image.shape[1]/self.screen_sections):
            image[:,i:i+5] = 0  
        self.display_state_image(image)      
        for i in range(self.screen_sections):
            if center[0] < image.shape[1] / (self.screen_sections * (i + 1)) and\
                center[0] > (image.shape[1] / self.screen_sections) * i:
                return i

        return self.screen_sections 
    '''

    def display_state_image(self, image):
        cv2.imshow("State", image)
        cv2.waitKey(25)  



