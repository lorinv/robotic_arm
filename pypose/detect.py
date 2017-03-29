import cv2
import numpy as np

cap = cv2.VideoCapture(1)
for i in range(10):
	ret, img = cap.read()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of green color in HSV
lower_green = np.array([60,0,100])
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



#for i in range(len(contours)):
#print contours[i]
#	print "hi"
m = cv2.moments(contours[max_contour])
x = m['m10'] /  m['m00']
y = m['m01'] /  m['m00']
print x
print y

grid = img.copy()
for i in range(0, img.shape[0], img.shape[0]/8):
    grid[i:i+5,:] = 0
for i in range(0, img.shape[1], img.shape[1]/8):
    grid[:,i:i+5] = 0
cv2.imshow("grid", grid)
cv2.waitKey(0)

cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), 4)
cv2.drawContours(img, contours, max_contour, (0,0,255), 3)
cv2.imshow("mask", mask)
cv2.imshow("img", img)
cv2.waitKey(0)

