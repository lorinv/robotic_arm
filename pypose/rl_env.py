from driver import Driver
from ax12 import *
import numpy
import random
import cv2
import itertools
import numpy as np
import time

DEBUG = False#True
TERMINAL_THRESHOLD_SIZE = .8

class Arm_Test:

    def __init__(self, move_const=10):
        self.env = Arm_Env(grid_dim=(8,8), num_motors=3)

        self.actions = []
        self.Q = {}
        for combination in itertools.product([0,move_const,-move_const], repeat=3):
            self.actions.append(combination)

        for state in range(self.env.num_states):
            for action in self.actions:
                self.Q[(state,action)] = random.uniform(0, 1) 

    def prompt_action(self):
        observation = 0
        done = 0
        reward = 0
        
        while 1:
            print "------------------------------------"
            print "Current State: %s" % str(observation)
            print "Reward: %s" % str(reward)
            print "Done: %s" % str(done)
            print "------------------------------------"            
            i = 0
            for action in self.actions:                                    
                    print "%d -- Action: %s \t Value: %f" % (i, str(action), self.Q[(observation,action)])
                    i += 1

            action = ""   
            while not action.isdigit():
                action = raw_input("Please Select Action: ")
            
                print "Chosen Action: %s" % str(self.actions[int(action)])
                observation, reward, done, info = self.env.step(self.actions[int(action)])     


class Arm_Env:

    def __init__(self, grid_dim=(16,16), num_motors=3):
        self.num_states = grid_dim[0] * grid_dim[1]
        self.num_actions = 3**num_motors                
        self.action_controller = Arm_Controller(num_motors=3)
        self.state_controller = Camera_State_Controller(clf_name="green_circle", grid_dim=grid_dim, camera_dev=2)

    def reset(self):
        self.action_controller.reset()  
        return self.step()  

    def step(self, action):             
        self.action_controller.take_action(action)        
        observation, area = self.state_controller.get_object_state()        
        reward = self.state_controller.distance_to_center(observation)        
        reward += area * 100
        done = area > TERMINAL_THRESHOLD_SIZE 
        info = None
        return sum(observation), reward, done, info

    def render(self):
        print "Look at the robot..."
        return 1

class Camera_State_Controller:

    def __init__(self, clf_name="green_circle", grid_dim=(16,16), camera_dev=0):
        if clf_name == "green_circle":
            self.clf = Green_Circle_Detector()
        elif clf_name == "wood_blocks":
            pass
        else:
            raise Exception('Error: CV classifier not found.')

        self.grid_dim = grid_dim
        self.cap = cv2.VideoCapture(camera_dev)
        self.img_shape = self.get_next_image().shape

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
        img = self.get_next_image()
        area, center = self.clf.detect(img)       
        grid_center = self.get_object_grid_center(center)
        #x1,y1,x2,y2 = self.clf.detect(img)       
        #x_center = (x2 - x1) / 2
        #y_center = (y2 - y1) / 2
        if DEBUG:
            grid = img.copy()
            for i in range(img.shape[1] / self.grid_dim[0], img.shape[1], img.shape[1]/self.grid_dim[0]):
                grid[i:i+5,:] = 0
            for i in range(img.shape[0] / self.grid_dim[0], img.shape[0], img.shape[0]/self.grid_dim[0]):
                grid[:,i:i+5] = 0
            cv2.imshow("grid", grid)
            cv2.waitKey(0)

        return grid_center, area

    #Returns the center of the object in terms of the grid space
    #Returns -1 if the object is not found
    def get_object_grid_center(self, center):    
        width, height, ch = self.img_shape
        per_row = width / self.grid_dim[0]
        per_col = height / self.grid_dim[1]
        x_cell = center[0] / per_row
        y_cell = center[1] / per_col


        
        return (x_cell, y_cell)

 


class Green_Circle_Detector(object):
    """docstring for ClassName"""
    def __init__(self):
        pass

    def detect(self, img):        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of green color in HSV
        lower_green = np.array([60,0,150])
        upper_green = np.array([80,254,254])
        mask = cv2.inRange(hsv, lower_green, upper_green)        

        im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = 0
        for i,c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > max_contour:
                max_area = area
                max_contour = i
        
        m = cv2.moments(contours[max_contour])
        x = int(m['m10'] /  m['m00'])
        y = int(m['m01'] /  m['m00'])
        print x
        print y
        if DEBUG:
            cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), 4)
            cv2.drawContours(img, contours, max_contour, (0,0,255), 3)
            cv2.imshow("mask", mask)
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return max_contour, (x,y)

class Arm_Controller:

    def __init__(self, num_motors=5):               
        self.motor_poses = []
        self.driver = Driver()
        #time.sleep(20)
        #for i in range(num_motors):                        
        #    self.driver.setReg(i+1, P_TORQUE_ENABLE, [0])
        #raw_input("Enter any value when ready.")
        print "Setting Motors..."
        time.sleep(10)
        #self.driver.setReg(1, P_PRESENT_POSITION_L, [0,0])
        '''
        for i in range(num_motors):            
            self.driver.setReg(i+1, P_TORQUE_ENABLE, [1])
            time.sleep(1)
            self.motor_poses.append(self.driver.getReg(i+1, P_PRESENT_POSITION_L, 2))
        print "Poses: %s" % str(self.motor_poses)        
        '''

    def reset(self):
        poses = [[]]

    def to_hl(self, value):        
        l = value & 255
        h = value >> 8

        return [l,h]

    def to_standard(self, values):
        l,h = values
        return (h<<8) | l

    def take_action(self, action):
        print "Former Poses: %s" % str(self.motor_poses)
        motor_vals = []        
        for i in range(len(self.motor_poses)):
            if i < len(action):
                val = self.to_standard(self.motor_poses[i])
                motor_vals.append(val)
        for i in range(len(self.motor_poses)):
            if i < len(action):
                motor_vals[i] += np.array(action[i])   

        for i in range(len(self.motor_poses)):
            if i < len(action):
                self.motor_poses[i] = self.to_hl(motor_vals[i])

        print "Poses: %s" % str(self.motor_poses)
        self.set_poses(self.motor_poses)         

    def set_poses(self, poses):
        if len(poses) != len(self.motor_poses):
            raise Exception('Error: incorrect numbebr of pose parameters passed.')

        for i in range(len(poses)):
            self.driver.setReg(i+1, P_PRESENT_POSITION_L, poses[i])     
            time.sleep(1)

    def get_poses(self):
        for i in range(num_motors):
            self.motor_poses[i] = self.driver.getReg(i+1, P_PRESENT_POSITION_L, 2)

        return self.motor_poses

    def check_load(self):
        loads = []
        for i in range(len(motor_poses)):
            loads.append(self.driver.getReg(i+1, P_PRESENT_LOAD_L, 2))

        return loads
    
class Sarsa_Lambda:

    def __init__(self):
        pass


if __name__ == '__main__':
    #a = Arm_Test()
    #a.prompt_action()
    #[[200, 1], [229, 1], [26, 2]]
    a = Arm_Controller(num_motors=1)
    a.set_poses([[0, 0]])
