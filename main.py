import os 
import sys 
import argparse
import glob
import time


import cv2
import numpy as np
from ultralytics import YOLO
from dronekit import connect, VehicleMode
from pymavlink import mavutil
# ============================================================
# ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file file (example: "runs/detect/train/weights/best.pt")', required=True)
parser.add_argument('--source', help='Image source: image file, folder, video file, or "usb0" / "picamera0"', required=True)
parser.add_argument('--thres', help='Minimum confidence threshold (example: "0.4")', default=0.5)
parser.add_argument('--resolution', help='Resolution WxH (example: "640x480")', default=None)
parser.add_argument('--record', help='Record video as demo1.avi. requires --resolution .', action='store_true')
parser.add_argument('--no-fly', help='Run detection only, do NOT send commands to Pixhawk (for testing)', action='store_true')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_threshold = float(args.thres)
user_res = args.resolution
record = args.record
no_fly = args.no_fly


# ============================================================
# DRONEKIT CONNECTION (TELEM2 = /dev/ttyAMA0)
# ============================================================

SERIAL_PORT = '/dev/ttyAMA0'
BAUD_RATE = 57600
vehicle = None
if no_fly:
    print('[DRONE] --no-fly mode: Pixhawk NOT connected. Detection only.')
else:
    print('[Drone] Connecting to Pixhawk on TELEM 2...')
    try:
        vehicle = connect(SERIAL_PORT, baud=BAUD_RATE, wait_ready=True)
        print(f'[Drone] connected to Pixhawk: {vehicle}')
    except Exception as e:
        print(f'[Drone] ERROR: Could not connect to Pixhawk; {e}')
        print('[DRONE] Tip: run with --no-fly to test detection without a drone.')
        sys.exit(0)

def set_guided_mode():
    if vehicle is None:
        return
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != 'GUIDED':
        print('[Drone] Waiting for GUIDED mode...')
        time.sleep(0.5)

def send_velocity(vx, vy, vz):
    if vehicle is None:
        return
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    vehicle.send_mavlink(msg)


def stop_drone():
    send_velocity(0, 0, 0)
# ============================================================
# YOLO MODEL 
# ============================================================

if not os.path.exists(model_path):
   print('ERROR: Model path is invalid or model was not found.')
   sys.exit(0)


model = YOLO(model_path, task='detect')
labels = model.names
print(f'[YOLO] Model loaded, Classes: {list(labels.values())}')


# ============================================================
# DISTANCE ESTIMATION SETUP
# ============================================================


REAL_WIDTH = 12.5 #cm (pre-set)(width of the object, for distance estimation)
KNOWN_DISTANCE = 50    # cm (pre-set)(distance at which you keep the object, for focal length calculation)
FOCAL_LENGTH   = 520   # px — set to 0 for calibration mode (the calibration code is the olddetectionscript.py file.)


# TARGET BEHAVIOUR
STOP_DISTANCE  = 30    # cm
FORWARD_SPEED  = 0.2   # m/s
LATERAL_SPEED  = 0.2   # m/s
DEAD_ZONE_X    = 60    # px
DEAD_ZONE_DIST = 5     # cm


def estimate_distance(perceived_width):
   return(REAL_WIDTH * FOCAL_LENGTH) / perceived_width

prev_distance = 0
alpha         = 0.7
# ============================================================
# SOURCE TYPE DETECTION
# ============================================================
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid.')
    sys.exit(0)

resW, resH = 640, 480
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

if record:
    if source_type not in ['video', 'usb', 'picamera']:
        print('Recording only works for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify --resolution to record.')
        sys.exit(0)
    record_name = 'demo1.avi'
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    for file in glob.glob(img_source + '/*'):
        if os.path.splitext(file)[1] in img_ext_list:
            imgs_list.append(file)
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# ============================================================
# STARTUP
# ============================================================
bbox_colors = [(164,120,87),(68,148,228),(93,97,209),(178,182,133),(88,159,106),
               (96,202,231),(159,124,168),(169,162,241),(98,118,150),(172,176,184)]

avg_frame_rate    = 0
frame_rate_buffer = []
fps_avg_len       = 200
img_count         = 0
FRAME_CENTER_X    = resW // 2

if not no_fly:
    set_guided_mode()

if FOCAL_LENGTH == 0:
    print('='*60)
    print('CALIBRATION MODE')
    print(f'Hold your flower ({REAL_WIDTH}cm wide) at exactly {KNOWN_DISTANCE}cm.')
    print('='*60)

# ============================================================
# MAIN LOOP
# ============================================================
while True:

    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed. Exiting.')
            sys.exit(0)
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('End of video. Exiting.')
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if frame is None or not ret:
            print('Camera disconnected. Exiting.')
            break

    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if frame is None:
            print('Picamera disconnected. Exiting.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results      = model(frame, verbose=False,)
    detections   = results[0].boxes
    object_count = 0

    best_detection = None
    best_conf      = 0.0

    for i in range(len(detections)):

        xyxy = detections[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        box_width = xmax - xmin
        cx        = (xmin + xmax) // 2

        classidx  = int(detections[i].cls.item())
        classname = labels[classidx]
        conf      = detections[i].conf.item()

        if conf > min_threshold:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10),
                          (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            # Calibration mode
            if FOCAL_LENGTH == 0 and box_width > 0:
                suggested_fl = (box_width * KNOWN_DISTANCE) / REAL_WIDTH
                print(f'[CALIBRATION] Box width: {box_width}px  |  Suggested FOCAL_LENGTH = {suggested_fl:.1f}')

            # Distance estimation
            distance = 0
            if FOCAL_LENGTH > 0 and box_width > 0:
                raw_distance  = estimate_distance(box_width)
                distance      = alpha * prev_distance + (1 - alpha) * raw_distance
                prev_distance = distance
                cv2.putText(frame, f'{distance:.1f} cm', (xmin, ymin-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            object_count += 1

            if conf > best_conf:
                best_conf      = conf
                best_detection = {'cx': cx, 'distance': distance, 'box_width': box_width}

    # ============================================================
    # FLIGHT CONTROL
    # ============================================================
    if not no_fly and vehicle is not None:

        if best_detection is None:
            stop_drone()
            cv2.putText(frame, 'No flower - holding', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cx       = best_detection['cx']
            distance = best_detection['distance']
            error_x  = cx - FRAME_CENTER_X

            vx = 0.0
            vy = 0.0

            if error_x > DEAD_ZONE_X:
                vy = LATERAL_SPEED
                direction_text = 'Correcting RIGHT'
            elif error_x < -DEAD_ZONE_X:
                vy = -LATERAL_SPEED
                direction_text = 'Correcting LEFT'
            else:
                vy = 0.0
                direction_text = 'Centered'

            if FOCAL_LENGTH > 0 and distance > 0:
                dist_error = distance - STOP_DISTANCE
                if dist_error > DEAD_ZONE_DIST:
                    vx = FORWARD_SPEED
                    approach_text = f'Approaching ({distance:.1f}cm)'
                elif dist_error < -DEAD_ZONE_DIST:
                    vx = -FORWARD_SPEED * 0.5
                    approach_text = f'Backing off ({distance:.1f}cm)'
                else:
                    vx = 0.0
                    approach_text = f'AT TARGET ({distance:.1f}cm)'
            else:
                approach_text = 'No distance (calibrate FL)'

            send_velocity(vx, vy, 0)

            cv2.putText(frame, direction_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
            cv2.putText(frame, approach_text, (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
            cv2.line(frame, (FRAME_CENTER_X, 0), (FRAME_CENTER_X, resH), (200,200,200), 1)
            cv2.arrowedLine(frame, (FRAME_CENTER_X, resH//2), (cx, resH//2), (0,255,255), 2)

    # ============================================================
    # HUD
    # ============================================================
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    if FOCAL_LENGTH == 0:
        cv2.putText(frame, 'CALIBRATION MODE: check terminal', (10,115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)
    if no_fly:
        cv2.putText(frame, '--no-fly: detection only', (10,115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 2)

    cv2.imshow('Flower Tracker', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey() if source_type in ['image', 'folder'] else cv2.waitKey(5)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    frame_rate_calc = 1.0 / (t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# ============================================================
# CLEANUP
# ============================================================
print(f'Average FPS: {avg_frame_rate:.2f}')
if not no_fly and vehicle is not None:
    stop_drone()
    vehicle.close()
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()

