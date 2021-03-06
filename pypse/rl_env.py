from driver import Driver
from ax12 import *
import numpy
import random
import cv2
import itertools
import numpy as np
import time
import copy

from camera_state_controller import Camera_State_Controller

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

    def __init__(self, grid_dim=(9,9), num_motors=3):
        """
        @grid_dim: nxn, where n is preferably some odd integer, such that the grid has a single center cell.
        """
        self.time_steps = 0
        self.num_states = grid_dim[0] * grid_dim[1]
        self.num_actions = 3**num_motors                
        self.action_controller = Arm_Controller(num_motors)
        self.state_controller = Camera_State_Controller(clf_name="green_circle", grid_dim=grid_dim, camera_dev=1)
        self.search_direction = 1
        self.motor_pos_stack = []
        
        #position information
        self.last_xy_position = numpy.array([0,0])
        self.current_xy_position = numpy.array([0,0])
        #goal defined as center of grid
        goal_x = grid_dim[0] / 2 + 1
        goal_y = grid_dim[1] / 2 + 1
        self.goal_xy = numpy.array([goal_x, goal_y])

        self.actions = []        
        for combination in itertools.product([0,MOVE_CONST,-MOVE_CONST], repeat=3):
            self.actions.append(combination)

    def reset(self):
        time.sleep(5)
        #self.action_controller.reset()          
        return self.step(0)  

    """
    Encapsulate our reward definition. This will likely change often depending on desired
    behavior/experimenation, so change this definition as ye please to test different reward constructions.
    
    @observation: 
    @xy_position: An (x,y) integer tuple representing the current object position on the grid.
    """
    def get_reward(self, observation, xy_position):
        #NOTE: returns reward as cosine-similarity between direction to the goal (center of image) and the direction of the last action.
        return self.goal_cossim()
    
        #NOTE: return this if we define reward as distance between object position and center of grid
        #return self.state_controller.distance_to_center(observation)

    def goal_cossim(self):
        """
        Assuming this object has stored the static center (x,y) point of the grid, and its
        last position, this calculates the cosine similarity between the displacement vector and the 
        position of the goal. In linear algebra terms, this is the cosine similarity of the the vector
        difference between the current (new) position and last position--call it v' = current_xy - last_xy--
        and the difference between the last position and the goal--call it g' = g_xy - last_xy.
        
        Defining a reward metric in terms of cosine-similarity wrt to the goal direction is nice because it is 
        constrained to the range [-1.0, 1.0], and gives a direct measure of the 'goodness' of an action's direction.
        
        NOTE: This function requires/assumes self.current_xy_position and self.last_xy_position have been updated.
        """
        v_prime = self.current_xy_position - self.last_xy_position
        g_prime = self.goal_xy - self.last_xy_position
        
        return self.cosine_similarity(v_prime, g_prime)
        
    def cosine_similarity(self, v1, v2):
        """
        Returns cosine similarity for two vectors v1, v2, both of which are 1xn numpy vectors.
        See wikipedia for definition of cos-sim. 
        @v1, @v2: Two numpy.array objects
        """
        return v1.dot(v2.T) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))

    """
    @action: list of motor actions {+c, -c, 0}
    
    returns: @observation, a unique state id for some state on nxn grid, where id in [0,n**2)
    """
    def step(self, action_id):   
        action = self.actions[action_id]
        self.time_steps += 1
        print 
        print
        print "TIME STEPS: %d" % self.time_steps
        print
        print
        
        if self.time_steps % 100 == 0:
            self.action_controller.take_a_break()
            time.sleep(120)
            #reset motor state to before break
            self.action_controller.set_poses(self.action_controller.motor_poses)
            time.sleep(2)

        #eg, (c, 0, -c)

        self.action_controller.take_action(action)
        observation, area, object_xy_position = self.state_controller.get_object_state()
        self.last_xy_position = self.current_xy_position
        self.current_xy_position = object_xy_position
        #@observation: center of the circle, 
        if area < MIN_AREA:
            observation, area, object_xy_position = self.revert_to_prev_state()
        print "\tReturned Area: %s" % str(area)
        print "Distance to Center: %s" % str(self.state_controller.distance_to_center(observation))
        reward = self.get_reward(observation, object_xy_position)
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

    def revert_to_prev_state(self):
        print "Lost the circle... Reverting Back"
        area = 0
        while area < MIN_AREA:
            self.action_controller.to_last_pose()        
            observation, area, object_xy_position = self.state_controller.get_object_state()
        return observation, area, object_xy_position
        

    def render(self):
        print "Look at the robot..."    
        return 1


class Arm_Controller:

    def __init__(self, num_motors=5):               
        self.motor_poses = []
        self.motor_pose_record = []
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

    def to_last_pose(self):
        print "Motor Pose Record: %s" % str(self.motor_pose_record)
        if len(self.motor_pose_record) > 1:            
            self.motor_poses = self.motor_pose_record.pop()#self.motor_pose_record[-1]
            print "Motor Poses"
            print self.motor_poses            
            self.set_poses(self.motor_poses)
            time.sleep(1)
            #raw_input("SETTING POSE!!")
            


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
        if self.to_standard(self.motor_poses[1]) < 450:
            self.motor_poses = self.get_poses()
            return False
        if self.to_standard(self.motor_poses[2]) < 450:
            self.motor_poses = self.get_poses()
            return False
        #if self.to_standard(self.motor_poses[2]) > 670:
        #    self.motor_poses = self.get_poses()
        #    return False     
        #if self.to_standard(self.motor_poses[1]) > 670:
        #    self.motor_poses = self.get_poses()
        #    return False   
        '''
        if self.to_standard(self.motor_poses[0]) > 850:
            self.motor_poses = self.get_poses()
            return False   
        if self.to_standard(self.motor_poses[0]) < 250:
            self.motor_poses = self.get_poses()
            return False   
        '''
        #if (self.to_standard(self.motor_poses[1]) + self.to_standard(self.motor_poses[2])) > 1400:
        #    self.motor_poses = self.get_poses()
        #    return False
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
            self.motor_poses[1] = self.to_hl(512)
            self.motor_pose_record.append(copy.copy(self.motor_poses))
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