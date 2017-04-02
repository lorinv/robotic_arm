from driver import Driver
from ax12 import *
import numpy
import random
import cv2
import itertools
import numpy as np
import time
import copy

DEBUG = True
TERMINAL_THRESHOLD_SIZE = 70000#39434
MIN_AREA = .001
MOVE_CONST = 20
MOTOR_SPEED = 25

class Arm_Test:

    def __init__(self, move_const=MOVE_CONST):
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
            max_action = 0
            max_action_value = 0
            max_i = 0
            for action in self.actions:                                    
                    print "%d -- Action: %s \t Value: %f" % (i, str(action), self.Q[(observation,action)])
                    if self.Q[(observation,action)] > max_action_value:
                        max_action = action
                        max_action_value = self.Q[(observation,action)]
                        max_i = i
                    i += 1

            print "Max Action: %d: %s" % (max_i,str(max_action))

            action = ""   
            while True:# not action.isdigit():
                #action = raw_input("Please Select Action: ")
                action = random.randint(0,26)
                #try:
                print "Chosen Action: %s" % str(self.actions[int(action)])
                observation, reward, done, info = self.env.step(self.actions[int(action)]) 
                if done:                    
                    observation, reward, done, info = self.env.reset()    
                    print("YOUR DONE!!")
                    print("YOUR DONE!!")
                    print("YOUR DONE!!")
                    print("YOUR DONE!!")
                    print("YOUR DONE!!")
                    print("YOUR DONE!!")
                    print("YOUR DONE!!")
                    print("YOUR DONE!!")
                    print("YOUR DONE!!")
                #except:
                #    pass


class Arm_Env:

    def __init__(self, grid_dim=(16,16), num_motors=3):
        self.time_steps = 0
        self.num_states = grid_dim[0] * grid_dim[1]
        self.num_actions = 3**num_motors                
        self.action_controller = Arm_Controller(num_motors)
        self.state_controller = Camera_State_Controller(clf_name="green_circle", grid_dim=grid_dim, camera_dev=1)
        self.search_direction = 1

        self.actions = []        
        for combination in itertools.product([0,MOVE_CONST,-MOVE_CONST], repeat=3):
            self.actions.append(combination)

    def reset(self):
        time.sleep(5)
        #self.action_controller.reset()          
        return self.step(0)  

    """
    @action: list of motor actions {+c, -c, 0}
    
    returns: @observation, a unique state id for some state on nxn grid, where id in [0,n**2)
    """
    def step(self, action_id):   
        action = self.actions[action_id]
        self.time_steps += 1
        
        if self.time_steps % 100 == 0:
            self.action_controller.take_a_break()
            time.sleep(60)
            #reset motor state to before break
            self.action_controller.set_poses(self.action_controller.motor_poses)
            time.sleep(2)

        #eg, (c, 0, -c)
        self.action_controller.take_action(action)        
        observation, area = self.state_controller.get_object_state()            
        #@observation: center of the circle, 
        if area < MIN_AREA:
            observation, area = self.find_object()
        print "\tReturned Area: %s" % str(area)        
        print "Distance to Center: %s" % str(self.state_controller.distance_to_center(observation))        
        reward = self.state_controller.distance_to_center(observation)        
        '''
        Linear Reward: maxd - d
        Neither w/ Margin -- 0

        Closer -- +1 
        Each Step -- -.1
        Goal State +10



        '''
        #reward += area * 100
        done = area > TERMINAL_THRESHOLD_SIZE 
        info = None
        return observation, reward, done, info

    def find_object(self):
        step = 0
        self.action_controller.take_action([MOVE_CONST*self.search_direction,0,0])
        observation, area = self.state_controller.get_object_state()
        while area < MIN_AREA:
            self.time_steps += 1      
            step += 1
            if step > 50:
                self.reset()
                step = 0
            search_pose = self.action_controller.to_standard(self.action_controller.motor_poses[0])
            if search_pose > 800 or search_pose < 300:                
                self.search_direction *= -1
                self.action_controller.take_action([MOVE_CONST*self.search_direction,0,0])
                self.action_controller.take_action([MOVE_CONST*self.search_direction,0,0])
            self.action_controller.take_action([MOVE_CONST*self.search_direction,0,0])

            observation, area = self.state_controller.get_object_state()

        return observation, area

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
        #self.cap = cv2.VideoCapture(camera_dev)
        #self.img_shape = self.get_next_image().shape

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
        area, center, img = self.clf.detect(None)       
        self.img_shape = img.shape
        grid_center = self.get_object_grid_center(center)
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

        return grid_center, area# / (img.shape[0] * img.shape[1]))

    #Returns the center of the object in terms of the grid space
    #Returns -1 if the object is not found
    def get_object_grid_center(self, center):    
        width, height, ch = self.img_shape    
        per_row = height / self.grid_dim[0]
        per_col = width / self.grid_dim[1]
        x_cell = center[0] / per_row
        y_cell = center[1] / per_col
        print x_cell
        print y_cell                
        
        return self.grid_dim[0] * int(y_cell) + int(x_cell)

class Green_Circle_Detector(object):
    """docstring for ClassName"""
    def __init__(self):
        pass

    def detect(self, img):      
        cap = cv2.VideoCapture(1)
        ret, img = cap.read()
        

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of green color in HSV
        lower_green = np.array([60,0,100])
        upper_green = np.array([80,254,254])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        cv2.imshow("Mask", mask)
        cv2.waitKey(5)        

        im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        
        if len(contours) < 1:
            return 0, (-1,-1), img

        max_area = 0
        max_contour = 0
        for i,c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > max_contour:
                max_area = area
                max_contour = i


        #print "\tArea: %s" % str(max_area) 
        if max_area < 20:
            return 0, (-1,-1), img

        #for i in range(len(contours)):
        #print contours[i]
        #   print "hi"
        m = cv2.moments(contours[max_contour])
        x = m['m10'] /  m['m00']
        y = m['m01'] /  m['m00']
        #print x
        #print y    

        cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), 4)
        cv2.drawContours(img, contours, max_contour, (0,0,255), 3)
                
        cap.release()
        return max_area, (x,y), img

class Arm_Controller:

    def __init__(self, num_motors=5):               
        self.motor_poses = []
        self.driver = Driver()   
        self.num_motors = num_motors     
        time.sleep(5)
        
        response = raw_input("Do you want to set the motors? (y/n) ")
        if 'y' in response:
            for i in range(num_motors):                        
                self.driver.setReg(i+1, P_TORQUE_ENABLE, [0])
            raw_input("Enter any value when ready.")
            print "Setting Motors..."

        #time.sleep(10)            
        for i in range(num_motors):            
            self.driver.setReg(i+1, P_TORQUE_ENABLE, [1])
            time.sleep(1)
            self.motor_poses.append(self.driver.getReg(i+1, P_PRESENT_POSITION_L, 2))    
        print "Poses: %s" % str(self.motor_poses)        
        self.original_pose = copy.copy(self.motor_poses)

    def reset(self):
        self.set_poses([self.to_hl(520),self.to_hl(520),self.to_hl(520)])
        time.sleep(5)
        self.get_poses()

    def to_hl(self, value):        
        l = value & 255
        h = value >> 8

        return [l,h]

    def to_standard(self, values):
        l,h = values
        return (h<<8) | l

    def check_pose(self):
        print "\nMotor Pose 0: %d" % self.to_standard(self.motor_poses[0])
        print "Motor Pose 1: %d" % self.to_standard(self.motor_poses[1])
        print "Motor Pose 2: %d" % self.to_standard(self.motor_poses[2])
        print "Combo: %d\n" % (self.to_standard(self.motor_poses[1]) + self.to_standard(self.motor_poses[2]))
        #raw_input("")
        if self.to_standard(self.motor_poses[1]) < 512:
            self.motor_poses = self.get_poses()
            return False
        if self.to_standard(self.motor_poses[2]) < 512:
            self.motor_poses = self.get_poses()
            return False
        if self.to_standard(self.motor_poses[2]) > 670:
            self.motor_poses = self.get_poses()
            return False     
        if self.to_standard(self.motor_poses[1]) > 670:
            self.motor_poses = self.get_poses()
            return False   
        '''
        if self.to_standard(self.motor_poses[0]) > 850:
            self.motor_poses = self.get_poses()
            return False   
        if self.to_standard(self.motor_poses[0]) < 250:
            self.motor_poses = self.get_poses()
            return False   
        '''
        if (self.to_standard(self.motor_poses[1]) + self.to_standard(self.motor_poses[2])) > 1400:
            self.motor_poses = self.get_poses()
            return False
        return True

    def take_action(self, action):        
        self.previous_pos = self.motor_poses
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
        if self.check_pose():
            self.set_poses(self.motor_poses)         

    def set_poses(self, poses):
        #if len(poses) != len(self.motor_poses):
        #    raise Exception('Error: incorrect numbebr of pose parameters passed.')

        for i in range(len(poses)):
            print "Motor: %d" % (i+1)      
            self.driver.setReg(i+1, 32, self.to_hl(MOTOR_SPEED))  
            #time.sleep(.5)         
            self.driver.setReg(i+1, P_GOAL_POSITION_L, poses[i])                 
            #time.sleep(.5)

    def take_a_break(self):
        #[[225, 1], [134, 3], [105, 1]]
        self.set_poses([[237, 1], [43, 3], [37, 1]])
        time.sleep(10)
        for i in range(self.num_motors):                        
                self.driver.setReg(i+1, P_TORQUE_ENABLE, [0])
                time.sleep(1)



    def get_poses(self):
        for i in range(self.num_motors):
            self.motor_poses[i] = self.driver.getReg(i+1, P_PRESENT_POSITION_L, 2)

        return self.motor_poses

    def check_load(self):
        loads = []
        for i in range(len(motor_poses)):
            loads.append(self.driver.getReg(i+1, P_PRESENT_LOAD_L, 2))

        return loads

    def check_safe_gaurds(self):
        pass

    
class Sarsa_Lambda:

    def __init__(self):
        pass


if __name__ == '__main__':
    a = Arm_Test()
    a.prompt_action()    
    #[[200, 1], [229, 1], [26, 2]]
    #a = Arm_Controller(num_motors=3)
    #print a.get_poses()
    #a.take_a_break()