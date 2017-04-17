import sys
sys.path.insert(0, 'hardware')

from driver import Driver
from ax12 import *
from time import sleep
MOTOR_SPEED = 35
MAX_LIMIT = 750
MIN_LIMIT = 350
NO_MOVE = 0
GO_LEFT = 1
GO_RIGHT = 2
GO_UP = 3
GO_DOWN = 4
BASE_MOTOR_STEP = 30
WRIST_MOTOR_STEP = 10

class ArmActions:

    def __init__(self):        
        self.driver = Driver()
        sleep(10)  
        self.setup_motors()
        self.actions = [[-BASE_MOTOR_STEP,0],[BASE_MOTOR_STEP,0],[0,WRIST_MOTOR_STEP],[0,-WRIST_MOTOR_STEP]] 
        self.num_actions = len(self.actions)   
        self.inital_pose = []
        for i in range(4):  
            self.inital_pose.append(self.driver.getReg(i+1, P_PRESENT_POSITION_L, 2))
        print self.inital_pose   

    def setup_motors(self, num_motors=4):
        response = raw_input("Do you want to set the motors? (y/n) ")
        if 'y' in response:
            for i in range(num_motors):                        
                self.driver.setReg(i+1, P_TORQUE_ENABLE, [0])
            raw_input("Enter any value when ready.")
            print "Setting Motors..."  

        for i in range(num_motors):            
            self.driver.setReg(i+1, P_TORQUE_ENABLE, [1])  

    def get_num_actions(self):
        return self.num_actions

    def take_action(self, action_id, state):
        if action_id == GO_RIGHT:                
            if state == 0 or state == 5 or state == 15 or state == 20:
                return
        if action_id == GO_LEFT:                
            if state == 4 or state == 9 or state == 14 or state == 19:
                return
        if action_id == GO_DOWN: 
            if state == 0 or state == 1 or state == 2 or state == 3 or state == 4:
                return
        if action_id == GO_UP: 
            if state == 20 or state == 21 or state == 22 or state == 23 or state == 24:
                return    

        self.set_motors(action_id)
    
    def set_motors(self, action_id):        
        self.driver.setReg(1, 32, self.to_hl(MOTOR_SPEED))        
        motor_pose = self.driver.getReg(1, P_PRESENT_POSITION_L, 2)     
        new_pose = self.to_standard(motor_pose) + self.actions[action_id][0]             
        if self.check_pose_limits(new_pose, motor=1):
            self.driver.setReg(1, 30, self.to_hl(new_pose))        

        self.driver.setReg(4, 32, self.to_hl(MOTOR_SPEED))        
        motor_pose = self.driver.getReg(4, P_PRESENT_POSITION_L, 2)     
        new_pose = self.to_standard(motor_pose) + self.actions[action_id][1]               
        if self.check_pose_limits(new_pose, motor=4):
            self.driver.setReg(4, 30, self.to_hl(new_pose))        

    def check_pose_limits(self, pose, motor):
        print "Motor: %d" % motor
        print "Pose: %d" % pose
        if motor == 1:
            if pose > MAX_LIMIT or pose < MIN_LIMIT:
                print "HITTING END OF RANGE OF MOTION!"
                return False
            else:
                return True
        '''
        if motor == 4:
            if pose > MAX_LIMIT or pose < MIN_LIMIT:
                return False
            else:
                return True
        '''
        return True


    def reset(self):
        for i in range(4):
            self.driver.setReg(i+1, 30, self.inital_pose[i])        
        sleep(3)        

    def to_hl(self, value):        
        l = value & 255
        h = value >> 8

        return [l,h]

    def to_standard(self, values):
        l,h = values
        return (h<<8) | l