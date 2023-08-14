import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

VERSION = "0.1.0"
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

          
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
	self.stopped = True


parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Sound file location
object_sounds = {
    'person': "person 1.wav',
    'bicyle': "bicyle 1.wav',
}

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
objects = {
    'person': {
        'left': 'person left 1.wav',
        'right': 'person right 1.wav',
        'centre': 'person centre 1.wav'
    },
    'bicycle': {
        'left': 'bicycle left 1.wav',
        'right': 'bicycle right 1.wav',
        'centre': 'bicycle centre 1.wav'
    },
    'car': {
        'left': 'car left 1.wav',
        'right': 'car right 1.wav',
        'centre': 'car centre 1.wav'
    },
    'motorcycle': {
        'left': 'motorcycle left 1.wav',
        'right': 'motorcycle right 1.wav',
        'centre': 'motorcycle centre 1.wav'
    },
    'airplane': {
        'left': 'airplane left 1.wav',
        'right': 'airplane right 1.wav',
        'centre': 'airplane centre 1.wav'
    },
    'bus': {
        'left': 'bus left 1.wav',
        'right': 'bus right 1.wav',
        'centre': 'bus centre 1.wav'
    },
    'train': {
        'left': 'train left 1.wav',
        'right': 'train right 1.wav',
        'centre': 'train centre 1.wav'
    },
    'truck': {
        'left': 'truck left 1.wav',
        'right': 'truck right 1.wav',
        'centre': 'truck centre 1.wav'
    },
    'boat': {
        'left': 'boat left 1.wav',
        'right': 'boat right 1.wav',
        'centre': 'boat centre 1.wav'
    },
    'traffic light': {
        'left': 'traffic light left 1.wav',
        'right': 'traffic light right 1.wav',
        'centre': 'traffic light centre 1.wav'
    },
    'fire hydrant': {
        'left': 'fire hydrant left 1.wav',
        'right': 'fire hydrant right 1.wav',
        'centre': 'fire hydrant centre 1.wav'
    },
    'stop sign': {
        'left': 'stop sign left 1.wav',
        'right': 'stop sign right 1.wav',
        'centre': 'stop sign centre 1.wav'
    },
    'parking meter': {
        'left': 'parking meter left 1.wav',
        'right': 'parking meter right 1.wav',
        'centre': 'parking meter centre 1.wav'
    },
    'bench': {
        'left': 'bench left 1.wav',
        'right': 'bench right 1.wav',
        'centre': 'bench centre 1.wav'
    },
    'bird': {
        'left': 'bird left 1.wav',
        'right': 'bird right 1.wav',
        'centre': 'bird centre 1.wav'
    },
    'cat': {
        'left': 'cat left 1.wav',
        'right': 'cat right 1.wav',
        'centre': 'cat centre 1.wav'
    },
    'dog': {
        'left': 'dog left 1.wav',
        'right': 'dog right 1.wav',
        'centre': 'dog centre 1.wav'
    },
    'horse': {
        'left': 'horse left 1.wav',
        'right': 'horse right 1.wav',
        'centre': 'horse centre 1.wav'
    },
    'sheep': {
        'left': 'sheep left 1.wav',
        'right': 'sheep right 1.wav',
        'centre': 'sheep centre 1.wav'
    },
    'cow': {
        'left': 'cow left 1.wav',
        'right': 'cow right 1.wav',
        'centre': 'cow centre 1.wav'
    },
    'elephant': {
        'left': 'elephant left 1.wav',
        'right': 'elephant right 1.wav',
        'centre': 'elephant centre 1.wav'
    },
    'bear': {
        'left': 'bear left 1.wav',
        'right': 'bear right 1.wav',
        'centre': 'bear centre 1.wav'
    },
    'zebra': {
        'left': 'zebra left 1.wav',
        'right': 'zebra right 1.wav',
        'centre': 'zebra centre 1.wav'
    },
    'giraffe': {
        'left': 'giraffe left 1.wav',
        'right': 'giraffe right 1.wav',
        'centre': 'giraffe centre 1.wav'
    },
    'backpack': {
        'left': 'backpack left 1.wav',
        'right': 'backpack right 1.wav',
        'centre': 'backpack centre 1.wav'
    },
    'umbrella': {
        'left': 'umbrella left 1.wav',
        'right': 'umbrella right 1.wav',
        'centre': 'umbrella centre 1.wav'
    },
    'handbag': {
        'left': 'handbag left 1.wav',
        'right': 'handbag right 1.wav',
        'centre': 'handbag centre 1.wav'
    },
    'tie': {
        'left': 'tie left 1.wav',
        'right': 'tie right 1.wav',
        'centre': 'tie centre 1.wav'
    },
    'suitcase': {
        'left': 'suitcase left 1.wav',
        'right': 'suitcase right 1.wav',
        'centre': 'suitcase centre 1.wav'
    },
    'frisbee': {
        'left': 'frisbee left 1.wav',
        'right': 'frisbee right 1.wav',
        'centre': 'frisbee centre 1.wav'
    },
    'skis': {
        'left': 'skis left 1.wav',
        'right': 'skis right 1.wav',
        'centre': 'skis centre 1.wav'
    },
    'snowboard': {
        'left': 'snowboard left 1.wav',
        'right': 'snowboard right 1.wav',
        'centre': 'snowboard centre 1.wav'
    },
    'sports ball': {
        'left': 'sports ball left 1.wav',
        'right': 'sports ball right 1.wav',
        'centre': 'sports ball centre 1.wav'
    },
    'kite': {
        'left': 'kite left 1.wav',
        'right': 'kite right 1.wav',
        'centre': 'kite centre 1.wav'
    },
    'baseball bat': {
        'left': 'baseball bat left 1.wav',
        'right': 'baseball bat right 1.wav',
        'centre': 'baseball bat centre 1.wav'
    },
    'baseball glove': {
        'left': 'baseball glove left 1.wav',
        'right': 'baseball glove right 1.wav',
        'centre': 'baseball glove centre 1.wav'
    },
    'skateboard': {
        'left': 'skateboard left 1.wav',
        'right': 'skateboard right 1.wav',
        'centre': 'skateboard centre 1.wav'
    },
    'surfboard': {
        'left': 'surfboard left 1.wav',
        'right': 'surfboard right 1.wav',
        'centre': 'surfboard centre 1.wav'
    },
    'tennis racket': {
        'left': 'tennis racket left 1.wav',
        'right': 'tennis racket right 1.wav',
        'centre': 'tennis racket centre 1.wav'
    },
    'bottle': {
        'left': 'bottle left 1.wav',
        'right': 'bottle right 1.wav',
        'centre': 'bottle centre 1.wav'
    },
    'wine glass': {
        'left': 'wine glass left 1.wav',
        'right': 'wine glass right 1.wav',
        'centre': 'wine glass centre 1.wav'
    },
    'cup': {
        'left': 'cup left 1.wav',
        'right': 'cup right 1.wav',
        'centre': 'cup centre 1.wav'
    },
    'fork': {
        'left': 'fork left 1.wav',
        'right': 'fork right 1.wav',
        'centre': 'fork centre 1.wav'
    },
    'knife': {
        'left': 'knife left 1.wav',
        'right': 'knife right 1.wav',
        'centre': 'knife centre 1.wav'
    },
    'spoon': {
        'left': 'spoon left 1.wav',
        'right': 'spoon right 1.wav',
        'centre': 'spoon centre 1.wav'
    },
    'bowl': {
        'left': 'bowl left 1.wav',
        'right': 'bowl right 1.wav',
        'centre': 'bowl centre 1.wav'
    },
    'banana': {
        'left': 'banana left 1.wav',
        'right': 'banana right 1.wav',
        'centre': 'banana centre 1.wav'
    },
    'apple': {
        'left': 'apple left 1.wav',
        'right': 'apple right 1.wav',
        'centre': 'apple centre 1.wav'
    },
    'sandwich': {
        'left': 'sandwich left 1.wav',
        'right': 'sandwich right 1.wav',
        'centre': 'sandwich centre 1.wav'
    },
    'orange': {
        'left': 'orange left 1.wav',
        'right': 'orange right 1.wav',
        'centre': 'orange centre 1.wav'
    },
    'broccoli': {
        'left': 'broccoli left 1.wav',
        'right': 'broccoli right 1.wav',
        'centre': 'broccoli centre 1.wav'
    },
    'carrot': {
        'left': 'carrot left 1.wav',
        'right': 'carrot right 1.wav',
        'centre': 'carrot centre 1.wav'
    },
    'hot dog': {
        'left': 'hot dog left 1.wav',
        'right': 'hot dog right 1.wav',
        'centre': 'hot dog centre 1.wav'
    },
    'pizza': {
        'left': 'pizza left 1.wav',
        'right': 'pizza right 1.wav',
        'centre': 'pizza centre 1.wav'
    },
    'donut': {
        'left': 'donut left 1.wav',
        'right': 'donut right 1.wav',
        'centre': 'donut centre 1.wav'
    },
    'cake': {
        'left': 'cake left 1.wav',
        'right': 'cake right 1.wav',
        'centre': 'cake centre 1.wav'
    },
    'chair': {
        'left': 'chair left 1.wav',
        'right': 'chair right 1.wav',
        'centre': 'chair centre 1.wav'
    },
    'couch': {
        'left': 'couch left 1.wav',
        'right': 'couch right 1.wav',
        'centre': 'couch centre 1.wav'
    },
    'potted plant': {
        'left': 'potted plant left 1.wav',
        'right': 'potted plant right 1.wav',
        'centre': 'potted plant centre 1.wav'
    },
    'bed': {
        'left': 'bed left 1.wav',
        'right': 'bed right 1.wav',
        'centre': 'bed centre 1.wav'
    },
    'dining table': {
        'left': 'dining table left 1.wav',
        'right': 'dining table right 1.wav',
        'centre': 'dining table centre 1.wav'
    },
    'toilet': {
        'left': 'toilet left 1.wav',
        'right': 'toilet right 1.wav',
        'centre': 'toilet centre 1.wav'
    },
    'tv': {
        'left': 'tv left 1.wav',
        'right': 'tv right 1.wav',
        'centre': 'tv centre 1.wav'
    },
    'laptop': {
        'left': 'laptop left 1.wav',
        'right': 'laptop right 1.wav',
        'centre': 'laptop centre 1.wav'
    },
    'mouse': {
        'left': 'mouse left 1.wav',
        'right': 'mouse right 1.wav',
        'centre': 'mouse centre 1.wav'
    },
    'remote': {
        'left': 'remote left 1.wav',
        'right': 'remote right 1.wav',
        'centre': 'remote centre 1.wav'
    },
    'keyboard': {
        'left': 'keyboard left 1.wav',
        'right': 'keyboard right 1.wav',
        'centre': 'keyboard centre 1.wav'
    },
    'cell phone': {
        'left': 'cell phone left 1.wav',
        'right': 'cell phone right 1.wav',
        'centre': 'cell phone centre 1.wav'
    },
    'microwave': {
        'left': 'microwave left 1.wav',
        'right': 'microwave right 1.wav',
        'centre': 'microwave centre 1.wav'
    },
    'oven': {
        'left': 'oven left 1.wav',
        'right': 'oven right 1.wav',
        'centre': 'oven centre 1.wav'
    },
    'toaster': {
        'left': 'toaster left 1.wav',
        'right': 'toaster right 1.wav',
        'centre': 'toaster centre 1.wav'
    },
    'sink': {
        'left': 'sink left 1.wav',
        'right': 'sink right 1.wav',
        'centre': 'sink centre 1.wav'
    },
    'refrigerator': {
        'left': 'refrigerator left 1.wav',
        'right': 'refrigerator right 1.wav',
        'centre': 'refrigerator centre 1.wav'
    },
    'book': {
        'left': 'book left 1.wav',
        'right': 'book right 1.wav',
        'centre': 'book centre 1.wav'
    },
    'clock': {
        'left': 'clock left 1.wav',
        'right': 'clock right 1.wav',
        'centre': 'clock centre 1.wav'
    },
    'vase': {
        'left': 'vase left 1.wav',
        'right': 'vase right 1.wav',
        'centre': 'vase centre 1.wav'
    },
    'scissors': {
        'left': 'scissors left 1.wav',
        'right': 'scissors right 1.wav',
        'centre': 'scissors centre 1.wav'
    },
    'teddy bear': {
        'left': 'teddy bear left 1.wav',
        'right': 'teddy bear right 1.wav',
        'centre': 'teddy bear centre 1.wav'
    },
    'hair drier': {
        'left': 'hair drier left 1.wav',
        'right': 'hair drier right 1.wav',
        'centre': 'hair drier centre 1.wav'
    },
    'toothbrush': {
        'left': 'toothbrush left 1.wav',
        'right': 'toothbrush right 1.wav',
        'centre': 'toothbrush centre 1.wav'
    }
}

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Play sound
            if object_name in object_sounds:
                sound_path = object_sounds[object_name]
                subprocess.run(['aplay', sound_path])
            else:
                print("No audio files found: ", object_name)

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
