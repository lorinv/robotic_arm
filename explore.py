from controller import ArmController
from time import sleep
import random
import cv2
import time

class Explore(object):
	"""docstring for ClassName"""
	def __init__(self):
		self.controller = ArmController()
		pass

	def start(self):
		cap = cv2.VideoCapture(1)
		found_object = False
		self.controller.set_joint(4, 900)	
		self.controller.set_joint(2,512)
		self.controller.set_joint(3,512)		
		position = self.controller.set_joint(1,0)	
		for i in range(800):
			position = self.controller.set_joint(1,position + 1)
			if position % 10 == 0:
				sleep(1)
				ret,img = cap.read()
				if img is not None:
					cv2.imshow("Window", img)
					cv2.waitKey(25)
					cv2.imwrite("PositiveSamples/pos_%s.jpg" % str(time.time()), img)					
			sleep(.01)
		for i in range(800):
			position = self.controller.set_joint(1 ,position - 1)
			if position % 10 == 0:
				sleep(1)
				ret,img = cap.read()
				if img is not None:
					cv2.imshow("Window", img)
					cv2.waitKey(25)
					cv2.imwrite("PositiveSamples/pos_%s.jpg" % str(time.time()), img)					
			sleep(.01)
			
		cap.close()

	def get_training_data(self, num_iters=10):	
		random.seed()	
		position1 = 0
		position4 = 0
		boolean = 1
		cap = cv2.VideoCapture(1)
		for i in range(num_iters):
			while position1 < 900:
					temp4 = position4
					position1 = self.controller.set_joint(1, position1 + 1)
					position4 = self.controller.set_joint(4, position4 + boolean)	
					if temp4 == position4 or random.randint(0,1000) < 10:
						boolean *= -1
					if position1 % 20 == 0:
						sleep(1)
						ret,img = cap.read()
						if img is not None:
							cv2.imshow("Window", img)
							cv2.waitKey(25)
							cv2.imwrite("NegativeSamples/neg_%s.jpg" % str(time.time()), img)					
			while position1 > 100:
					temp4 = position4
					position1 = self.controller.set_joint(1, position1 - 1)
					position4 = self.controller.set_joint(4, position4 - boolean)	
					if temp4 == position4 or random.randint(0,1000) < 10:
						boolean *= -1	
					if position1 % 20 == 0:
						sleep(1)
						ret,img = cap.read()
						if img is not None:
							cv2.imshow("Window", img)
							cv2.waitKey(25)
							cv2.imwrite("NegativeSamples/neg_%s.jpg" % str(time.time()), img)	

		cap.close()

if __name__ == "__main__":  
	e = Explore()
	#e.get_training_data()
	e.start()

		
