import numpy
from controller import ArmController

class Arm_Env:

	def __init__(self):
		self.observation_space.n = 10
		self.action_space.n = 27
		self.controller = ArmController()	

	def reset(self):
		self.controller.set_joint(4, 900)   
        self.controller.set_joint(2,512)
        self.controller.set_joint(3,512) 
        self.controller.set_joint(1,0) 

	def step(self, action):
		self.controller = set_joint(self, motor_id, position)
		return observation, reward, done, info

	def render(self):
		pass

class Arm_State:

	def __init__(self):
		pass

	


if __name__ == '__main__':
	pass

