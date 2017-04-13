import numpy as np

class SlidingWindow:

	def __init__(self, image_set, window_size=(20,40), stride=10):
		self.image_count = 0
		self.image_set = image_set
		self.current_image = self.image_set[0]			
		self.window_x_pos = 0
		self.window_y_pos = 0
		self.stride = stride
		self.window_size = window_size

	def get_next_window(self):
		self.window_x_pos += self.stride
		self.window_y_pos += self.stride
		rows, cols = self.window_size[1], self.window_size[0]
		image_rows, image_cols = self.current_image.shape
		self.check_last_column(self.window_x_pos, cols, image_cols)		
		self.check_last_row(self.window_y_pos, rows, image_rows)
		if self.current_image is not None:
			return self.current_image[self.window_y_pos:rows, self.window_x_pos:cols]
		else:
			return None
	
	def check_last_column(self, x, cols, image_cols):
		if x+cols > image_cols:
			self.window_x_pos = 0					

	def check_last_row(self, y, rows, image_rows):
		if y+rows > image_rows:
			self.image_count += 1
			self.window_y_pos = 0
			self.window_x_pos = 0	

	def check_last_image(self):		
		if len(self.image_set) - 1 < self.image_count:
			self.current_image = None
		else
			self.current_image = self.image_set[self.image_count]

	def reset_images(self):
		self.image_count = 0
		self.current_image = self.image_set[self.image_count]





