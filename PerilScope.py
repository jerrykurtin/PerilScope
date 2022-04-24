import cv2
import numpy as np
import serial
from elements.yolo import OBJ_DETECTION
from collections import deque
import time

import gc
from tkinter import DISABLED

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
LIDAR_WINDOW = 7   # Frames
LIDAR_INTERVAL = 5
SPD_THRESH = 0.5
ALERT_THRESH = 1.5
# DEBUG
SPD_THRESH = 0
ALERT_THRESH = 0.5
WIDTH, HEIGHT = 1280, 720
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_TYPE = cv2.VideoWriter_fourcc(*'XVID')

debug = False
save = True
model = "yolov5s.pt"
filename = "context_tests2_{:.2f}.mp4".format(time.time())

# attention_obj = set(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'fire hydrant', 'stop sign', 'parking meter', 'cat', 'dog', 'backpack', 'handbag', 'suitcase', 'skateboard', 'bottle', 'knife'])

important_obj = set(['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'skateboard', 'knife', 'scissors', 'bottle', 'backpack', 'handbag', 'suitcase'])
dangerous_obj = set(['knife', 'scissors'])
 
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

def direction_context(distances):
    """
    distances: (distance, time) a list of tuples
    fps: how fast the camera is sensing
    returns: tuple of (context, speed)
        context: 0, 1, 2 (not a problem, approaching, or leaving)
    """

    length = len(distances)-1
    total_spd = 0
    for idx in range(1, len(distances)):
        if (distances[idx][1] - distances[idx-1][1]) == 0:
            # print("dropping frame,", distances)
            length -= 1
            continue
        total_spd += (distances[idx][0]-distances[idx-1][0])/(distances[idx][1]-distances[idx-1][1])
    total_spd /= len(distances)-1
    total_spd /= -100   # scale and reverse

    if total_spd < -1 * SPD_THRESH:
        return (total_spd, 2)
    elif total_spd > SPD_THRESH:
        return (total_spd, 1)
    else:
        return (total_spd, 0) 

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

# Image Capture and save
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
if save:
    out = cv2.VideoWriter(f"videos/{filename}", VIDEO_TYPE, 9, (WIDTH, HEIGHT))
if cap.isOpened():
    
    # ----------------------------Main Loop-----------------------------------
    window_handle = cv2.namedWindow("PerilScope Live Feed", cv2.WINDOW_AUTOSIZE)
    # variables
    st_time = time.time()
    flag = 60
    frames = 0
    distances = deque()
    dsum = 0
    null_ctr = 0
    speed, direct = 0, 0
    tick = 0

    while cv2.getWindowProperty("PerilScope Live Feed", 0) >= 0:
        tick += 1
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
        if tick % LIDAR_INTERVAL == 0:
            window_start = time.time()

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
        if distance < LIDAR_INTERVAL or distance > 1200:
            # manage queue
            null_ctr += 1
            if null_ctr > 20:
                distances.clear()
                dsum = 0
            d_msg = ("----", "---")
            # d_msg = "Distance: {} ft. Confidence: {}".format("----", "---")
        else:
            null_ctr = 0
            if tick % LIDAR_INTERVAL == 0:
                # manage distance tracking
                distances.append((distance, window_start))
                dsum += distance
                if len(distances) > LIDAR_WINDOW:
                    dsum -= distances.popleft()[0]
            d_msg = (distance * 0.0328084, "High" if confident_lidar else "Low")
            # d_msg = "Distance: {:.2f} ft. Confidence: {}".format(distance * 0.0328084, "High" if confident_lidar else "Low")
            

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
                frame = cv2.putText(frame, f'{label}', (xmin,ymin+3), FONT, 0.75, color, 1, cv2.LINE_AA)

        # Determine context
        panik = False
        if len(distances) >= LIDAR_WINDOW and tick % LIDAR_INTERVAL == 0:
            speed, direct = direction_context(list(distances))
        if abs(speed) > SPD_THRESH:
            panik = True
        if d_msg[1] == "---":
            d_msg = "Distance: {} ft. Speed: {} mph. Confidence: {}".format(d_msg[0], d_msg[0], d_msg[1])
        else:
            d_msg = "Distance: {:.2f} ft. Speed: {:.2f} mph. Confidence: {}".format(d_msg[0], speed * 2.23694, d_msg[1])
        
        obj_list = [obj['label'] for obj in objs]
        in_frame = "Object"
        person, vehicle, small_vehicle, weapon = False, False, False, False
        if ret and direct > 0:
            if 'person' in obj_list:
                person = True
            # small vehicle
            if 'bicycle' in obj_list or 'skateboard' in obj_list:
                small_vehicle = True
            if 'car' in obj_list or 'motorcycle' in obj_list or 'bus' in obj_list or 'train' in obj_list or 'truck' in obj_list:
                vehicle = True
            if 'knife' in obj_list or 'scissors' in obj_list:
                weapon = True
        if person:
            in_frame = "Person"
        if weapon:
            panik = True
            in_frame = "Armed " + in_frame.lower()
        if vehicle:
            if not person:
                in_frame = "Vehicle"
            else:
                in_frame += " in vehicle"
        elif small_vehicle:
            if not person:
                in_frame = "Small vehicle"
            in_frame += " in small vehicle"

        direction = "front"


        # Write on display
        msg = f"Elapsed: {time.time()-st_time:.0f} sec"
        frame = cv2.putText(frame, msg, (10, HEIGHT-50), FONT, 0.75, MAROON, 2, cv2.LINE_AA) 
        msg = f"Average FPS: {frames/(time.time()-st_time):.1f}"
        frame = cv2.putText(frame, msg, (10, HEIGHT-30), FONT, 0.75, MAROON, 2, cv2.LINE_AA) 

        # Write context stats
        if direct > 0 and len(distances) >= LIDAR_WINDOW:
            # aware mode
            msg = "{} {} {}".format(in_frame, "approaching" if direct == 1 else "retreating", direction)
            # alert mode
            if abs(speed) > SPD_THRESH:
                msg = msg.upper()
                frame = cv2.rectangle(frame, (0,0), (WIDTH, HEIGHT), RED, 5)
            textSize = cv2.getTextSize(msg, FONT, 1, 2)[0]
            textX, textY = int((WIDTH - textSize[0])/2), int((HEIGHT-textSize[1])/2)
            frame = cv2.putText(frame, msg, (textX, 35), FONT, 1.0, MAROON, 3, cv2.LINE_AA) 
        msg = d_msg
        textSize = cv2.getTextSize(msg, FONT, 1, 2)[0]
        textX, textY = int((WIDTH - textSize[0])/2), int((HEIGHT-textSize[1])/2)
        frame = cv2.putText(frame, msg, (textX, 65), FONT, 1.0, MAROON, 3, cv2.LINE_AA) 

        if save:
            out.write(frame)
        cv2.imshow("CSI Camera", frame)
        keyCode = cv2.waitKey(30)
        if keyCode == ord('q'):
            break

    # Closing commands
    ser.close()
    cap.release()
    cv2.destroyAllWindows()
    del Object_detector
    gc.collect()
    print("Successfully exited program")
else:
    print("Unable to open camera")
