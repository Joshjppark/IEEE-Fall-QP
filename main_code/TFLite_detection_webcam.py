######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py

# Import packages
import os
import argparse
import cv2
import math
import numpy as np
import sys
import struct
import time
import bluepy
import importlib.util
import pyaudio
from threading import Thread
from python2bluetooth import send2bluetooth

hm10_address = "30:E2:83:8D:78:20"

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        #(self.grabbed, cv2.Flip(self.frame, flipMode=-1)) = self.stream.read()
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
        

class ClapDetector():
    
    def __init__(self):        
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=44100,
                            input=True,
                            frames_per_buffer=1024
                            )
        self.clap_threshold = 2
        self.noisycount = 3 + 1 # adding one so that the program does not register a clap on first listen() call
        self.quietcount = 0
        self.detected = False
        # self.increment_clap = False
    
    
    def get_rms(self, block):
        # RMS amplitude is the squart root of the mean of the squares of the ampltiude
        count = len(block)/2 # we divide by 2 to get the number of samples, since each sample is written with 2 bytes
        format = f"{int(count)}h"
        int_block = struct.unpack(format, block) # converts data into 16 bit integers
        
        # iterate over the block
        sum_squares = 0.0
        for sample in int_block:
            # sample is a signed 16 bit integer in +/- 32768
            n = sample * (1.0/32768.0) # normalizes the data
            sum_squares += n*n
            
        return math.sqrt(sum_squares/count)

    def start(self):
	# Start the thread that listens for claps from audio stream
        Thread(target=self.update,args=()).start()
        return self
    
    def stop(self):
        self.stream.close()

    def listen(self):
        block = self.stream.read(1024)
        amplitude = self.get_rms(block) * 100
        if amplitude > self.clap_threshold:
            # the block is noisy:
            self.quietcount = 0
            self.noisycount += 1
        else:
            # block is quiet
            if 1 <= self.noisycount <= 3:
                # a clap is detected if it is noisy for a little bit
                self.clapDetected()
            self.noisycount = 0
            self.quietcount += 1

        
    def clapDetected(self):
        self.detected = True
        # print(f'Clap')

    def update(self):
        leds_on = False
        num_claps = 0
        print('Start')
        start_time = time.time()
        try:
            while True:
                self.listen()
                now = time.time()
                if(self.detected):
                    self.detected = False
                    num_claps += 1

                    if(num_claps == 2):
                        print('DOUBLE Clap')
                        num_claps = 0

                        if leds_on: # leds are on; turn them off
                            send2bluetooth(hm10_address, 'H')
                            leds_on = False
                        else: # leds are off; turn them on
                            send2bluetooth(hm10_address, 'L')
                            leds_on = True

                    else:
                        start_time = time.time()
                        print('SINGE Clap')
                if(round(now - start_time, 1)) > 1:
                    num_claps = 0
        except KeyboardInterrupt:
            clap.stop()
            print('Finished')


# Define and parse input arguments
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
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--videofeed', help='Play video or not', default=False)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
PLAY_VIDEO = args.videofeed

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
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

CWD_PATH = os.getcwd() # Get path to current working directory
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME) # Path to .tflite file, which contains the model that is used for object detection
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME) # Path to label map file

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
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
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

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()


videostream = VideoStream(resolution=(imW,imH),framerate=30).start() # Initialize video stream
clap = ClapDetector().start() #Initalize audio stream
time.sleep(1)
if PLAY_VIDEO:
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL) # Create window

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
previous_detected = False
while True:
    t1 = cv2.getTickCount() # Start timer (for calculating frame rate)
    frame1 = videostream.read() # Grab frame from video stream

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
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if int(classes[i]) == 0:
            continue

        if((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):            
            previous_detected = True
            send2bluetooth(hm10_address, 'D')
            
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW))) 
            xcenter = xmin + (int(round((xmax - xmin) / 2)))
            ycenter = ymin + (int(round((ymax - ymin) / 2)))
            
            if PLAY_VIDEO:
                # Get bounding box coordinates and draw box
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                
                # Draw label
                label = '%s: %d%%' % ('person', int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin + labelSize[0], label_ymin + baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                # Draw circle in     
                cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)

            # Print info
            print(f'Object {i}: Person {scores[i]*100}% at ({xcenter}, {ycenter})')

        else:
            # person not detected
            if previous_detected:
                send2bluetooth(hm10_address, 'N')
                print('Person not detected')
                previous_detected = False

    if PLAY_VIDEO:
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
    else:
        print(f'FPS: {round(frame_rate_calc, 2)}')

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