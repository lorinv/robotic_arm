import cv2
import itertools
import numpy as np
import time
import copy

#from object_detectors import Green_Contour_Detector
from selective_search_hog import BlockDetector


DEBUG = True

class Camera_State_Controller:

    def __init__(self, clf_name="green_circle", grid_dim=(16,16), camera_dev=0):
        if clf_name == "green_circle":
            self.clf = BlockDetector()
        elif clf_name == "wood_blocks":
            pass
        else:
            raise Exception('Error: CV classifier not found.')

        self.grid_dim = grid_dim
        #self.cap = cv2.VideoCapture(camera_dev)
        #self.img_shape = self.get_next_image().shape

    #returns distance between @point and img center
    def distance_to_center(self, point):
        width, height, ch = self.img_shape
        img_cen = (width / 2, height / 2)
        distance = np.linalg.norm(np.array(img_cen)-np.array(point))
        return distance

    def get_next_image(self):       
        for i in range(10):
            ret, img = self.cap.read()
        if img is None:
            raise Exception('Error: Failed to read image.')

        return img

    #Returns the center of the object in the image
    #Returns -1 if the object is not found
    def get_object_state(self):
        #img = self.get_next_image()
        center, area, img = self.clf.detect()       
        self.img_shape = img.shape
        grid_center, xy_center = self.get_object_grid_position(center)
        #x1,y1,x2,y2 = self.clf.detect(img)       
        #x_center = (x2 - x1) / 2
        #y_center = (y2 - y1) / 2
        if DEBUG:
            grid = img.copy()
            for i in range(0, img.shape[0], img.shape[0]/self.grid_dim[0]):
                grid[i:i+5,:] = 0
            for i in range(0, img.shape[1], img.shape[1]/self.grid_dim[0]):
                grid[:,i:i+5] = 0
            cv2.imshow("grid", grid)
            cv2.waitKey(5)

        return grid_center, area, xy_center # / (img.shape[0] * img.shape[1]))

    #Returns the position of the object in terms of the grid space. The first item returned represents
    #the grid cell radix id, the second is an (x,y) numpy.array representing the absolute position in fixed grid space.
    #Returns -1 if the object is not found
    def get_object_grid_position(self, center):
        width, height, ch = self.img_shape    
        per_row = height / self.grid_dim[0]
        per_col = width / self.grid_dim[1]
        x_cell = center[0] / per_row
        y_cell = center[1] / per_col
        print x_cell
        print y_cell                
        
        return self.grid_dim[0] * int(y_cell) + int(x_cell), np.array([x_cell,y_cell])

