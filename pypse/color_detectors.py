import cv2
import numpy as np

class Green_Circle_Detector(object):
    """docstring for ClassName"""
    def __init__(self):
        pass

    def detect(self, img):      
        cap = cv2.VideoCapture(1)
        while 1:            
            ret, img = cap.read()
            img = cv2.normalize(img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # define range of green color in HSV
            lower_green = np.array([60,0,100])
            upper_green = np.array([80,254,254])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            cv2.imshow("Mask", mask)
            cv2.waitKey(5)        

            #im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            distance_between_circles = 10
            circles = cv2.HoughCircles(mask,cv.CV_HOUGH_GRADIENT,1,distance_between_circles)#,param1=50,param2=30,minRadius=0,maxRadius=25)
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(mask,(i[0],i[1]),i[2],(0,255,0),2)

            cv2.imshow('detected circles',mask)
            if cv2.waitKey(0) > 0:
                cv2.destroyAllWindows()
                break
                
        
        if len(contours) < 1:
            return 0, (-1,-1), img
        '''
        max_area = 0
        max_contour = 0
        for i,c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > max_contour:
                max_area = area
                max_contour = i
        '''


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

class Green_Contour_Detector(object):
    """docstring for ClassName"""
    def __init__(self):
        pass

    def detect(self, source=1):      
        cap = cv2.VideoCapture(1)
        average_x = 0
        average_y = 0
        count = 0
        average_area = 0
        while 1:
            count += 1
            ret, img = cap.read()        
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)            

            # define range of green color in HSV
            lower_green = np.array([65,195,120])
            upper_green = np.array([82,256,240])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            kernel = np.ones((10,10),np.float32)/25
            mask = cv2.filter2D(mask,-1,kernel)
            lower_green = np.array([1])
            upper_green = np.array([255])
            mask = cv2.inRange(mask, lower_green, upper_green)

            cv2.imshow("Mask", mask)
            cv2.waitKey(5)        

            im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
            
            if len(contours) < 1:
                return 0, (-1,-1), img
            
            max_area = 0
            max_contour = 0
            for i,c in enumerate(contours):
                print "hi"
                area = cv2.contourArea(c)
                if area > max_contour:
                    max_area = area
                    max_contour = i
            
            #print "\tArea: %s" % str(max_area) 
            #if max_area < 20:
            #    return 0, (-1,-1), img

            #for i in range(len(contours)):
            #print contours[i]
            #   print "hi"
            m = cv2.moments(contours[max_contour])
            x = m['m10'] /  m['m00']
            y = m['m01'] /  m['m00']
            #print x
            #print y    

            average_x += x
            average_y += y
            average_area += max_area
            a = 2
            if count % a == 0:
                average_area /= a
                average_x /= a
                average_y /= a
                cv2.circle(img, (int(average_x), int(average_y / a)), 10, (0, 0, 255), 4)
                cv2.drawContours(img, contours, max_contour, (0,0,255), 3)
                average_x = average_y = 0

            
                cv2.imshow("Green Circles", img)
                cv2.waitKey(25)
                break

        cap.release()
        return average_area, (average_x,average_y), img

# import the necessary packages
import argparse
import cv2
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
image = None
 
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, image
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        print "Color: %s" % str(image[y][x])
        '''
        #cropping = True
 
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
 
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)
    '''

def get_color_at_click():    
    # load the image, clone it, and setup the mouse callback function    
    global image
    cap = cv2.VideoCapture(1)
    ret, image = cap.read()         
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
     
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        ret, image = cap.read()         
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clone = image.copy()
        cv2.imshow("image", image)
        key = cv2.waitKey(25) & 0xFF
        '''
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
     
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
        '''
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    '''
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
    '''
     
    # close all open windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    get_color_at_click()
    #detector = Green_Contour_Detector()
    #detector.detect(sys.argv[1])