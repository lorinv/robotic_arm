from ml_detectors import Selective_Search, HOG_Color_Hist_Classifier

import cv2
import numpy as np
import time

ROI_SHAPE = (35,35)

class BlockDetector:

    def __init__(self):
        pass        
    
    def detect(self, image=None):
        if image is None:
            cap = cv2.VideoCapture(1) 
            ret, image = cap.read()
            cap.release()
        if image is None:
            raise "Error: Invalid Image Receieved"
        ss = Selective_Search()
        regions = ss.generate_regions(image)
        roi_set = ss.get_roi_set(image, regions)
        
        classifier = HOG_Color_Hist_Classifier()
        classifier.load_classifier("computer_vision/hog_color_hist_classifier.clf")

        for i, roi in enumerate(roi_set):
            roi_original_area = roi.shape[0] * roi.shape[1]
            if roi.shape[0] * roi.shape[1] < image.shape[0] * image.shape[1] * .8:
                roi = cv2.resize(roi, ROI_SHAPE, interpolation = cv2.INTER_CUBIC)       
                label = classifier.predict_image_label(roi)
                if label == 1:
                    x, y, w, h = list(regions[i])
                    center_cood = (x + (w / 2), y + (h / 2))
                    print "Center Coordinates: %s" % str(center_cood)
                    cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)
                    return center_cood, roi_original_area, image
        
        return [-1,-1], -1, image
        '''
        print "Label: %s" % str(label)          
        cv2.imshow("Image", image)
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        '''


        '''
        true_label = int(raw_input("Select Label: "))
        if true_label != label[0]:
            print "Writing file..."
            if true_label == 1:             
                cv2.imwrite("recycle_bin/pos/" + str(time.time()) + "_" + str(i) + ".jpg", roi)             
            elif true_label == 0:               
                cv2.imwrite("recycle_bin/neg/" + str(time.time()) + "_" + str(i) + ".jpg", roi)             
        '''


    


if __name__ == '__main__':
    import sys
    detect(cv2.imread(sys.argv[1]))
