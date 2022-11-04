import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import PIL
import pyttsx3
import espeak
from collections import defaultdict
from io import StringIO
from PIL import Image
import RPi.GPIO as GPIO
import time
import tensorflow.compat.v1 as tf
import cv2


#GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
#set GPIO Pins
GPIO_TRIGGER = 4
GPIO_ECHO = 27
GPIO_BUZZ = 17

def most_frequent(index_array):
    a = np.bincount(index_array).argmax()
    
    return a 
 
#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
GPIO.setup(GPIO_BUZZ, GPIO.OUT)
def distance():
    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    GPIO.output(GPIO_BUZZ, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance


cap = cv2.VideoCapture(0)
sys.path.append("..")

from object_detection.utils import label_map_util
 
#from utils import visualization_utils as vis_util
from object_detection.utils import visualization_utils as vis_util 
 
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/ieee/Desktop/tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
 
NUM_CLASSES = 90

        
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
 
 
detection_graph = tf.compat.v1.get_default_graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
  
#detection_graph = tf.compat.v1.GraphDef() 
#with detection_graph.as_default():
# od_graph_def = tf.GraphDef()
with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:

        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
 
 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#engine = pyttsx3.init()
if __name__ == '__main__':
	try:
		with detection_graph.as_default():
		    
		    with tf.Session(graph=detection_graph) as sess:
		        while True:
		            ret, image_np = cap.read()
		            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		            image_np_expanded = np.expand_dims(image_np, axis=0)
		            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		            # Each box represents a part of the image where a particular object was detected.
		            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		            # Each score represent how level of confidence for each of the objects.
		            # Score is shown on the result image, together with the class label.
		            scores = detection_graph.get_tensor_by_name('detection_scores:0')
		            classes = detection_graph.get_tensor_by_name('detection_classes:0')
		            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		            # Actual detection.
		            (boxes, scores, classes, num_detections) = sess.run(
		            [boxes, scores, classes, num_detections],
		            feed_dict={image_tensor: image_np_expanded})
		            # Visualization of the results of a detection.
		            vis_util.visualize_boxes_and_labels_on_image_array(
		                image_np,
		                np.squeeze(boxes),
		                np.squeeze(classes).astype(np.int32),
		                np.squeeze(scores),
		                category_index,
		                use_normalized_coordinates=True,
		                line_thickness=8)
		     
		            cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
		            #espeak.synth("this is a " + image_tensor)
		            
		            index_array = np.squeeze(classes).astype(np.int32) 
		            index_to_say = most_frequent(index_array)
		            #print('awaited result',index_to_say)
		            final_list = [d['name'] for d in categories]
		            #print(final_list)
		            toSay = final_list[index_to_say-1]
		            say = 'espeak "{0}"'.format(toSay)
		            os.system(say)
		            print( np.squeeze(classes).astype(np.int32))
		              DELETE THIS COMMENT WHEN DONE WORKING 
		            toSay = 'espeak "{0}"'.format()
		            os.system(toSay)
		            
		            os.system("espeak " + image_tensor[0])
		            espeak.init()
		            speaker = espeak.Espeak()
		            speaker.say("Hello world")
		            os.system('espeak "'+ image_tensor)
		            dist = distance()
		            print("Measured Distance = %.1f cm" % dist)
		            if (dist < 100 ):
		                GPIO.output(GPIO_BUZZ,True)
		                print ("Measured Distance  below 100")
		            time.sleep(1)
		            if cv2.waitKey(1) == ord('q'):
		                cv2.destroyAllWindows()
		                
	except KeyboardInterrupt:
		print("Measurement stopped by User")
		GPIO.cleanup()
