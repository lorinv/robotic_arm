import sys
sys.path.insert(0, 'hardware')

from driver import Driver
from ax12 import *
from time import sleep
MOTOR_SPEED = 35
MAX_LIMIT = 800
MIN_LIMIT = 300
NO_MOVE = 0
TURN_LEFT = 1
TURN_RIGHT = 2

class ArmActions:

    def __init__(self):
        self.num_actions = 3    
        self.driver = Driver()
        sleep(5)  
        self.setup_motors()
        self.actions = [0,-70,70] 
        self.inital_pose = []
        for i in range(3):
            self.inital_pose.append(self.driver.getReg(i+1, P_PRESENT_POSITION_L, 2))
        print self.inital_pose   

    def setup_motors(self, num_motors=4):
        response = raw_input("Do you want to set the motors? (y/n) ")
        if 'y' in response:
            for i in range(num_motors):                        
                self.driver.setReg(i+1, P_TORQUE_ENABLE, [1])
            raw_input("Enter any value when ready.")
            print "Setting Motors..."  

        for i in range(num_motors):            
            self.driver.setReg(i+1, P_TORQUE_ENABLE, [1])  

    def get_num_actions(self):
        return self.num_actions

    def take_action(self, action_id, state):
        if state == 0:
            if action_id == TURN_LEFT:
                return
        elif state == 2:
            if action_id == TURN_RIGHT:
                return

        self.set_motors(action_id)
    
    def set_motors(self, action_id):        
        self.driver.setReg(1, 32, self.to_hl(MOTOR_SPEED))
        motor_pose = self.driver.getReg(1, P_PRESENT_POSITION_L, 2)     
        new_pose = self.to_standard(motor_pose) + self.actions[action_id]               
        if self.check_pose_limits(new_pose):
            self.driver.setReg(1, 30, self.to_hl(new_pose))        

    def check_pose_limits(self, pose):
        if pose > MAX_LIMIT or pose < MIN_LIMIT:
            return False
        else:
            return True

    def reset(self):
        for i in range(3):
            self.driver.setReg(1, 32, self.inital_pose[i])        
        sleep(2)        

    def to_hl(self, value):        
        l = value & 255
        h = value >> 8

        return [l,h]

    def to_standard(self, values):
        l,h = values
        return (h<<8) | l