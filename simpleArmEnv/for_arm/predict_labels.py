import tensorflow as tf
import os
import json
from train import build_forward
from utils.train_utils import add_rectangles
import cv2
import argparse

DEBUG = True

'''
This uses the TensorFlow package called TensorBox.
It utilizes Google's Inception Network to detect objects.
'''
class TensorBoxPrediction:

    def __init__(self):
        self.min_conf = 0.2
        self.tau = 0.25
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        hypes_file = 'for_arm/classifier/hypes.json'
        with open(hypes_file, 'r') as f:
            self.H = json.load(f)
        self.weight_args = "for_arm/classifier/save.ckpt-6000"
        self.setup_network()

    def setup_network(self):
        tf.reset_default_graph()
        self.x_in = tf.placeholder(tf.float32, name='x_in', shape=[self.H['image_height'], self.H['image_width'], 3])
        if self.H['use_rezoom']:
            self.pred_boxes, pred_logits, self.pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(self.H, tf.expand_dims(self.x_in, 0), 'test', reuse=None)
            grid_area = self.H['grid_height'] * self.H['grid_width']
            self.pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * self.H['rnn_len'], 2])), [grid_area, self.H['rnn_len'], 2])
            if self.H['reregress']:
                self.pred_boxes = self.pred_boxes + pred_boxes_deltas
        else:
            self.pred_boxes, pred_logits, self.pred_confidences = build_forward(self.H, tf.expand_dims(self.x_in, 0), 'test', reuse=None)
        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer()) #Do I need this?
        saver.restore(self.sess, self.weight_args)        

    def detect(self, img):
        original_shape = img.shape
        img = cv2.resize(img, (self.H["image_width"], self.H["image_height"]), interpolation = cv2.INTER_CUBIC)
        feed = {self.x_in: img}
        (np_pred_boxes, np_pred_confidences) = self.sess.run([self.pred_boxes, self.pred_confidences], feed_dict=feed)
 
        new_img, rects = add_rectangles(self.H, [img], np_pred_confidences, np_pred_boxes,
                                        use_stitching=True, rnn_len=self.H['rnn_len'], min_conf=self.min_conf, tau=self.tau, show_suppressed=False)
        rect = None
        output_rects = []
        for r in rects:
            rect = {"x":r.x1, "y":r.y1, "width":r.x2 - r.x1, "height":r.y2 - r.y1}
            rect = self.to_full_size_detections(original_shape, new_img.shape, rect)
            output_rects.append(rect)
            break
        

        if DEBUG:    
            cv2.imshow("test", new_img)
            cv2.waitKey(25)            

        if rect is not None:
            return (int(rect['x'] + (rect['width'] / 2.0)), int(rect['y'] + (rect['height'] / 2.0))), new_img
        else:
            return (-1,-1), new_img
        #return output_rects

    def to_full_size_detections(self, full_image_shape, mod_image_shape, label):
        f_rows, f_cols = full_image_shape[:2]
        m_rows, m_cols = mod_image_shape[:2]
        rows_percentage = float(f_rows) / m_rows
        cols_percentage = float(f_cols) / m_cols
        label['x'] *= rows_percentage
        label['y'] *= cols_percentage
        label['width'] *= rows_percentage
        label['height'] *= cols_percentage

        return label 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',nargs='*', required=True, type=str)
    args = parser.parse_args()
    tb = TensorBoxPrediction()
    for path in args.image:
        image = cv2.imread(path)
        detection_rects = tb.predict(image)
        with open(path+".labels", 'w') as f:
            json.dump(detection_rects, f)

if __name__ == '__main__':
    main()
