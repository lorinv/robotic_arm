from driver import Driver
from ax12 import *
import numpy

DEBUG = True
TERMINAL_THRESHOLD_SIZE = .8

class Arm_Env:

	def __init__(self):		
		self.action_controller = Arm_Controller(4)
		self.state_controller = Camera_State_Controller(clf="green_circle", grid_dim=(16,16), camera_dev=1)

	def reset(self):
		self.action_controller.reset()		

	def step(self, action):		
		observation = self.state_controller.get_object_grid_center()
		reward = self.state_controller.distance_to_center(observation)
		size = self.state_controller.get_object_size()
		reward += size * 100
		done = size > TERMINAL_THRESHOLD_SIZE 
		info = None
		return observation, reward, done, info

	def render(self):
		print "Look at the robot..."
		return 1

class Camera_State_Controller:

	def __init__(self, clf_name="green_circle", grid_dim=(16,16), camera_dev=0):
		if clf_name == "green_circle":
			clf = Green_Circle_Detector()
		elif clf_name == "wood_blocks":
			pass
		else:
			raise Exception('Error: CV classifier not found.')

		self.grid_dim = grid_dim
		self.cap = cv2.VideoCapture(camera_dev)
		self.img_shape = self.get_next_image().shape

	def distance_to_center(self, point):
		return distance

	def get_next_image(self):
		ret, img = self.cap.read()
		if img is None:
			raise Exception('Error: Failed to read image.')

		return img

	#Returns the center of the object in the image
	#Returns -1 if the object is not found
	def get_object_center(self):
		img = self.get_next_image()
		x1,y1,x2,y2 = clf.detect(img)		
		x_center = (x2 - x1) / 2
		y_center = (y2 - y1) / 2

		return (x_center, y_center)

	#Returns the center of the object in terms of the grid space
	#Returns -1 if the object is not found
	def get_object_grid_center(self):
		center = self.get_object_grid_center()
		width, height, ch = self.img_shape
		per_row = width / grid_dim[0]
		per_col = height / grid_dim[1]
		x_cell = center[0] / per_row
		y_cell = center[1] / per_col
		
		return (x_cell, y_cell)

	#Returns the percentage of the image that the 
	#object fills
	def get_object_size(self):
		img = self.get_next_image()
		x1,y1,x2,y2 = clf.detect(img)	
		width, height, ch = self.img_shape	
		x_len = abs(x2 - x1) 
		y_len = abs(y2 - y1)
		x_size = float(x_len) / width
		y_size = float(y_len) / height

		return (x_size + y_size) / 2


class Green_Circle_Detector(object):
	"""docstring for ClassName"""
	def __init__(self):
		pass

	def detect(img):
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	    # define range of green color in HSV
	    lower_green = np.array([110,50,50])
	    upper_green = np.array([130,255,255])

	    # Threshold the HSV image to get only green colors
	    mask = cv2.inRange(hsv, lower_green, upper_green)
	    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	    if DEBUG:
	    	img_temp = img.copy()	    	
	    	print contours
	    	cv2.drawContours(img_temp, contours, 3, (0,255,0), 3)
	    	cv2.imshow("Show Contour", img_temp)
	    	cv2.waitKey(0)	    

	    return self.contour_to_rect(contour[0])

	def contour_to_rect(self, contour):
		return rect

class Arm_Controller:

	def __init__(self, num_motors=5):		
		self.motor_poses = []
		self.driver = Driver()
		for i in range(num_motors):
			self.motor_poses.append(self.driver.getReg(i+1, P_PRESENT_POSITION_L, 2))

	def reset(self):
		poses = [[]]

	def to_hl(self, value):
		pass

	def to_standard(self, values):
		pass

	def set_poses(self, poses):
		if len(poses) != len(self.motor_poses):
			raise Exception('Error: incorrect numbebr of pose parameters passed.')

		for i in range(len(poses)):
			self.driver.setReg(i+1, P_PRESENT_POSITION_L, poses[i])		

	def get_poses(self):
		for i in range(num_motors):
			self.motor_poses[i] = self.driver.getReg(i+1, P_PRESENT_POSITION_L, 2)

		return self.motor_poses

	def check_load(self):
		loads = []
		for i in range(len(motor_poses)):
			loads.append(self.driver.getReg(i+1, P_PRESENT_LOAD_L, 2))

		return loads
		








	


if __name__ == '__main__':
	pass

