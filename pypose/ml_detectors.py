import cv2
import numpy as np
from skimage.feature import hog
from skimage import color
from skimage.exposure import histogram
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

import selectivesearch

IS_COLOR_IMAGE = 3

class HOG_Color_Hist_Classifier:

    def __init__(self):
        self.clf = None     

    def compute_hog_features(self, image):   
        if len(image.shape) == IS_COLOR_IMAGE:
            image = color.rgb2gray(image)
        fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=False)    

        return fd

    def compute_color_histogram(self, image, num_bins=20):
        if len(image.shape) != IS_COLOR_IMAGE:
            raise "Error: Please pass color image"
        hist = histogram(image, nbins=num_bins)
        return hist

    def compute_linear_svm_classifier(self, X, y):
        if X.shape[0] != len(y):
            raise "Error: Number of labels do not match the number of images."
        clf = svm.LinearSVC()
        clf.fit(X, y)
        self.clf = clf
        return clf

    def compute_features(self, image):
        hog = self.compute_hog_features(image)
        color_hist = self.compute_hog_features(image)
        return np.hstack((hog, color_hist))

    def predict_image_label(self, image):        
        features = self.compute_features(image)
        return self.predict_feature_label(features)

    def predict_feature_label(self, features):
        if self.clf is None:
            raise "Error: classifier not yet computed"
        return self.clf.predict(features)

    def write_classifier(self, file_path):
        joblib.dump(self.clf, file_path) 
        print("Status: Classifier Saved.")

    def load_classifier(self, file_path):           
        self.clf = joblib.load(file_path)
        print("Status: Classifier Loaded.")

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return np.array(a)[p], np.array(b)[p]

    def compute_data_matrix(self, posRawSamples, negRawSamples, test_set_percentage=.1):
        posFeatures = classifier.compute_features(posRawSamples[0])
        negFeatures = classifier.compute_features(negRawSamples[0])
        for pos in posRawSamples[1:]:        
            posFeatures = np.vstack((posFeatures, classifier.compute_features(pos)))
        for neg in negRawSamples[1:len(posRawSamples)]:
            negFeatures = np.vstack((negFeatures, classifier.compute_features(neg)))

        labels = [1] * len(posFeatures) + [0] * len(posFeatures)

        data_matrix, labels = self.unison_shuffled_copies(np.vstack((posFeatures,negFeatures)), labels)
        return train_test_split(data_matrix, labels, test_size=test_set_percentage, random_state=42)

class Selective_Search:

    def __init__(self):
        pass

    def generate_regions(self, image):
        img_lbl, regions = selectivesearch.selective_search(
            image, scale=500, sigma=0.9, min_size=10)

        return self.exclude_bad_regions(regions)

    def exclude_bad_regions(self, regions):
        candidates = []
        for r in regions:   
        # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 1000:
                continue
            # distorted rects
            x, y, w, h = r['rect']
            if w / h > 1.2 or h / w > 1.2:
                continue        
            candidates.append(r['rect'])

        return candidates

    def get_roi_set(self, image, regions):
        roi_set = []
        for x,y,w,h in regions:
            if w > 20 and h > 20:
                roi_set.append(image[y:y+h,x:x+w])
        return roi_set

    def get_roi_square_set(self, image, regions):
        for x,y,w,h in regions:
            smaller_dim = min(w,h)
            if smaller_dim > 20:
                roi_set.append(image[y:y+smaller_dim,x:x+smaller_dim])

        return roi_set            


if __name__ == '__main__':
    import sys
    posRawSamples = []
    negRawSamples = []
    posFeatures = None
    negFeatures = None
    for path in sys.argv[1:]:
        image = cv2.imread(path)
        smaller_dim = min(image.shape[0],image.shape[1])
        image = image[0:smaller_dim, 0:smaller_dim]        
        image = cv2.resize(image, (35,35), interpolation = cv2.INTER_CUBIC)
        if "pos" in path:            
            print path
            posRawSamples.append(image)
        else:
            negRawSamples.append(image)

    classifier = HOG_Color_Hist_Classifier()
    X_train, X_test, y_train, y_test = classifier.compute_data_matrix(posRawSamples, negRawSamples[:len(posRawSamples)])
    classifier.compute_linear_svm_classifier(X_train, y_train)
    correct_labels = 0
    for i, feature in enumerate(X_test):        
        label = classifier.predict_feature_label(feature)
        print "Label: %s" % str(label)
        if label == y_test[i]:
            correct_labels += 1

    print "Classifier Accuracy: %s" % str(float(correct_labels)/len(y_test))
    classifier.write_classifier("hog_color_hist_classifier.clf")




