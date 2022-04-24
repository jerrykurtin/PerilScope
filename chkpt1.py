from tkinter import DISABLED
import cv2
import numpy as np
import serial
from elements.yolo import OBJ_DETECTION
import time
from collections import deque

# ----------------------variables and setup------------------------------------
# BGR
MAROON = (0, 0, 80)
RED = (0, 0, 255) 
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
# BLUE = (102, 255, 0)
BLUE2 = (0, 255, 102)

LIDAR_ATTEMPTS = 5
LIDAR_WINDOW = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
WIDTH, HEIGHT = 1280, 720

debug = False
model = "yolov5s.pt"

# attention_obj = set(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'fire hydrant', 'stop sign', 'parking meter', 'cat', 'dog', 'backpack', 'handbag', 'suitcase', 'skateboard', 'bottle', 'knife'])

important_obj = set(['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat', 'skateboard', 'knife', 'bottle', 'backpack', 'handbag', 'suitcase'])
dangerous_obj = set(['person', 'knife', 'scissors'])
 
Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush' ]

Object_colors = list(np.random.rand(80,3)*255)

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=WIDTH,
    display_height=HEIGHT,
    framerate=60,
    flip_method=2,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=2))

# ------------------------Model/Devices Setup-------------------------------

# LIDAR Module
ser = serial.Serial("/dev/ttyTHS1", 115200)
if not ser.is_open:
    ser.open()
if not ser.is_open:
    print("ERROR: unable to open lidar sensor")

# YOLO model
if not debug:
    Object_detector = OBJ_DETECTION('weights/{}'.format(model), Object_classes)

# Image Capture
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
if cap.isOpened():
    
    # ----------------------------Main Loop------------------------------------
    

    window_handle = cv2.namedWindow("PerilScope Live Feed", cv2.WINDOW_AUTOSIZE)
    # variables
    st_time = time.time()
    flag = 60
    frames = 0
    distances = deque()
    dsum = 0

    while cv2.getWindowProperty("PerilScope Live Feed", 0) >= 0:

        # exclude the first few frames
        if flag > 0: 
            flag -= 1
            frames -= 1
        if flag == 0:
            flag = -1
            st_time = time.time()
        
        # Read a frame
        frames += 1
        ret, frame = cap.read()

        # read/process lidar signal
        confident_lidar = False
        for _ in range(LIDAR_ATTEMPTS):
            recv = ser.read(9)
            ser.reset_input_buffer()

            if recv[0] == 0x59 and recv[1] == 0x59:
                distance = recv[2] + recv[3] * 256
                strength = recv[4] + recv[5] * 256
                # print('(', distance, ',', strength, ')')
                ser.reset_input_buffer()
            
                if strength >= 200:
                    confident_lidar = True
                    break

        # bad values are eliminated
        if distance < 10 or distance > 1200:
            d_msg = "Distance: {} ft. Confidence: {}".format("----", "---")
        else:
            
            d_msg = "Distance: {:.2f} ft. Confidence: {}".format(distance * 0.0328084, "High" if confident_lidar else "Low")
            

        # detection process
        if ret and not debug:
            objs = Object_detector.detect(frame)
            for obj in objs:
                # print(obj)
                label = obj['label']
                score = obj['score']
                if label in dangerous_obj:
                    color = RED
                elif label in important_obj:
                    color = YELLOW
                else:
                    continue
                    color = BLUE
                # color = Object_colors[Object_classes.index(label)]

                [(xmin,ymin),(xmax,ymax)] = obj['bbox']
                frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
                # frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin), FONT, 0.75, color, 1, cv2.LINE_AA)
                frame = cv2.putText(frame, f'{label}', (xmin,ymin), FONT, 0.75, color, 1, cv2.LINE_AA)


        # Write on display
        msg = f"Elapsed: {time.time()-st_time:.0f} sec"
        frame = cv2.putText(frame, msg, (10, 30), FONT, 0.75, MAROON, 2, cv2.LINE_AA) 
        msg = f"Average FPS: {frames/(time.time()-st_time):.1f}"
        frame = cv2.putText(frame, msg, (10, 50), FONT, 0.75, MAROON, 2, cv2.LINE_AA) 

        # Write context stats
        msg = "HUMAN APPROACHING FRONT LEFT"
        textSize = cv2.getTextSize(msg, FONT, 1, 2)[0]
        textX, textY = int((WIDTH - textSize[0])/2), int((HEIGHT-textSize[1])/2)
        frame = cv2.putText(frame, msg, (textX, 30), FONT, 1.0, MAROON, 3, cv2.LINE_AA) 
        msg = d_msg
        textSize = cv2.getTextSize(msg, FONT, 1, 2)[0]
        textX, textY = int((WIDTH - textSize[0])/2), int((HEIGHT-textSize[1])/2)
        frame = cv2.putText(frame, msg, (textX, 60), FONT, 1.0, MAROON, 3, cv2.LINE_AA) 


        cv2.imshow("CSI Camera", frame)
        keyCode = cv2.waitKey(30)
        if keyCode == ord('q'):
            break

    # Closing commands
    ser.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Successfully exited program")
else:
    print("Unable to open camera")
