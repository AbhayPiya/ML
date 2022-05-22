# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from utils import alertcheck
detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'frozen_graphs'
PATH_TO_CKPT_person = TRAINED_MODEL_DIR + '/persondetection.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS_person = TRAINED_MODEL_DIR + '/person.pbtxt'

PATH_TO_CKPT_mask = TRAINED_MODEL_DIR + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS_mask = TRAINED_MODEL_DIR + '/Labelmap.pbtxt'

NUM_CLASSES = 2

label_map_person = label_map_util.load_labelmap_person(PATH_TO_LABELS_person)
label_map_mask = label_map_util.load_labelmap_mask(PATH_TO_LABELS_mask)


categories_mask = label_map_util.convert_label_map_to_categories_mask(
    label_map_mask, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index_mask= label_map_util.create_category_index(categories_mask)

categories_person = label_map_util.convert_label_map_to_categories_person(
    label_map_person, max_num_classes=1, use_display_name=True)
category_index_person = label_map_util.create_category_index(categories_person)

a=b=0

# Load a frozen infererence graph into memory
def load_inference_graph_mask():

    # load frozen tensorflow model into memory
    print("> ====== Loading frozen graph into memory")
    detection_graph_mask = tf.Graph()
    with detection_graph_mask.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_mask, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess_mask = tf.Session(graph=detection_graph_mask)
    print(">  ====== Inference graph loaded.")
    return detection_graph_mask, sess_mask

def load_inference_graph_person():
    # load frozen tensorflow model into memory
    print("> ====== Loading frozen graph into memory")
    detection_graph_person = tf.Graph()
    with detection_graph_person.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_person, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess_person = tf.Session(graph=detection_graph_person)
    print(">  ====== Inference graph loaded.")
    return detection_graph_person, sess_person





def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes,im_width, im_height, image_np,Line_Position2,Orientation):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differentiate distances and detected boxes

    global a,b
    hand_cnt=0
    color = None
    color0 = (255,0,0)
    color1 = (0,50,255)
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            if classes[i] == 1:
                id = 'no mask'
            if classes[i]==2:
                id ='mask'
                avg_width = 3.0
            
            if i == 0:
                color = color0
            else:
                color = color1

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            dist = distance_to_camera(avg_width, focalLength, int(right-left))
            
            if dist:
                hand_cnt=hand_cnt+1           
            cv2.rectangle(image_np, p1, p2, color, 3, 1)

            cv2.putText(image_np,'person', (int(left)-40, int(top)-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(image_np, str(i)+': '+id, (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.putText(image_np, 'distance from camera: '+str("{0:.2f}".format(dist)+' inches'),
                       (int(im_width*0.65),int(im_height*0.9+30*i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)
           
            a=alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)
        if hand_cnt==0 :
            b=0
        else:
            b=1

            
    return a,b

# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph_mask, sess_mask):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph_mask.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph_mask.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph_mask.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph_mask.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph_mask.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess_mask.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)

def detect_objects_person(image_np, detection_graph_mask, sess_mask):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph_mask.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph_mask.get_tensor_by_name('detection_boxes:1')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph_mask.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph_mask.get_tensor_by_name('detection_classes:1')
    num_detections = detection_graph_mask.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess_mask.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
