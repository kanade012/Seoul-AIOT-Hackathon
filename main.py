import cv2
import cvzone
import torch
import numpy as np
import math
from ultralytics import YOLO
import time
import pygame  # 알람 사운드를 재생하기 위한 라이브러리

# Initialize pygame for sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('beep.mp3')  # 삐 소리를 담은 파일 경로 설정

# Load YOLOv8 models
person_model = YOLO('model/yolov8n.pt')
knife_model = YOLO('model/knife_model.pt')  # 칼 탐지 모델
pose_model = YOLO('model/yolov8n-pose.pt')

# Initialize video capture (webcam or video file)
cap = cv2.VideoCapture(0)

# Define keypoints for pose estimation
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Person tracking and knife detection data
person_data = {}
person_id_counter = 0
fall_duration_threshold = 1.0

knife_detection_times = {}
alerts = []
alert_spacing = 60
alert_expiry_time = 5

draw_layer = True

# Initial value for up/down counter
counter_value = 0

def get_person_id(person_boxes, person_data, threshold=50):
    global person_id_counter
    assigned_ids = []
    for box in person_boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        current_person_pos = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        matched_id = None
        for person_id, last_position in person_data.items():
            if 'position' in last_position:
                dist = np.linalg.norm(current_person_pos - last_position['position'])
                if dist < threshold:
                    matched_id = person_id
                    person_data[person_id]['position'] = current_person_pos
                    break

        if matched_id is None:
            person_id_counter += 1
            person_data[person_id_counter] = {'position': current_person_pos}
            matched_id = person_id_counter

        assigned_ids.append(matched_id)
    return assigned_ids

def detect_fall_with_duration(person_id, current_keypoints, person_data):
    try:
        neck = current_keypoints[5]
        hip = current_keypoints[11]
        left_ankle = current_keypoints[15]
        right_ankle = current_keypoints[16]

        neck_hip_diff = abs(neck[1] - hip[1])
        hip_ankle_diff = abs(hip[1] - min(left_ankle[1], right_ankle[1]))

        neck_hip_threshold = 100
        hip_ankle_threshold = 100

        if neck_hip_diff < neck_hip_threshold and hip_ankle_diff < hip_ankle_threshold:
            if 'fall_start_time' not in person_data[person_id]:
                person_data[person_id]['fall_start_time'] = time.time()
            else:
                fall_duration = time.time() - person_data[person_id]['fall_start_time']
                if fall_duration >= fall_duration_threshold:
                    return True
        else:
            if 'fall_start_time' in person_data[person_id]:
                del person_data[person_id]['fall_start_time']
    except (IndexError, TypeError) as e:
        print(f"Error in fall detection: {e}")
        return False

    return False

def get_next_alert_position(alerts):
    y_base = 50
    return (50, y_base + len(alerts) * alert_spacing)

def add_alert(alerts, alert_text, position, person_id=None):
    for existing_alert, existing_pos, existing_id, _ in alerts:
        if existing_alert == alert_text and existing_id == person_id:
            return
    alerts.append((alert_text, position, person_id, time.time()))

def remove_expired_alerts(alerts):
    current_time = time.time()
    return [(text, pos, obj_id, t) for text, pos, obj_id, t in alerts if current_time - t < alert_expiry_time]

# 알람 상태에 따라 노트북에서 소리 재생
def play_alert_sound(alerts):
    if len(alerts) > 0:
        alert_sound.play(maxtime=1000)  # 1초 동안만 소리 재생


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (980, 740))
    person_results = person_model(frame)
    knife_results = knife_model(frame)

    for person_info in person_results:
        person_boxes = person_info.boxes
        person_ids = get_person_id(person_boxes, person_data)

        for person_box, person_id in zip(person_boxes, person_ids):
            x1, y1, x2, y2 = person_box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = person_box.conf[0]
            class_detect = person_box.cls[0]
            class_detect = int(class_detect)
            conf = math.ceil(confidence * 100)

            if conf > 50 and class_detect == 0:
                width = x2 - x1
                height = y2 - y1

                if draw_layer:
                    cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                    cvzone.putTextRect(frame, f"Person {person_id}", [x1 + 8, y1 - 12], thickness=2, scale=2)

                with torch.no_grad():
                    pose_results = pose_model(frame)

                if pose_results and pose_results[0].keypoints is not None:
                    for pose_result in pose_results:
                        keypoints = pose_result.keypoints.xy[0]

                        if len(keypoints) < 6:
                            continue

                        if draw_layer:
                            for i, kp in enumerate(keypoints):
                                if i < len(keypoint_names):
                                    x, y = int(kp[0].item()), int(kp[1].item())
                                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                                    cv2.putText(frame, keypoint_names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 0, 0), 1)

                        if detect_fall_with_duration(person_id, keypoints, person_data):
                            alert_position = get_next_alert_position(alerts)
                            add_alert(alerts, f'Fall Detected: Person {person_id}', alert_position, person_id)

    for knife_info in knife_results:
        knife_boxes = knife_info.boxes
        for box in knife_boxes:
            x1_k, y1_k, x2_k, y2_k = box.xyxy[0]
            x1_k, y1_k, x2_k, y2_k = int(x1_k), int(y1_k), int(x2_k), int(y2_k)
            confidence_k = box.conf[0]
            class_detect_k = box.cls[0]
            class_detect_k = int(class_detect_k)
            conf_k = math.ceil(confidence_k * 100)

            if conf_k > 50 and class_detect_k == 0:
                width_k = x2_k - x1_k
                height_k = y2_k - y1_k

                knife_id = (x1_k, y1_k, x2_k, y2_k)
                if knife_id not in knife_detection_times:
                    knife_detection_times[knife_id] = {'start_time': time.time(), 'valid': False}

                detection_time = time.time() - knife_detection_times[knife_id]['start_time']

                if detection_time >= 0.5:
                    knife_detection_times[knife_id]['valid'] = True
                    if knife_detection_times[knife_id]['valid']:
                        if draw_layer:
                            cvzone.cornerRect(frame, [x1_k, y1_k, width_k, height_k], l=30, rt=6)
                            cvzone.putTextRect(frame, "Knife", [x1_k + 8, y1_k - 12], thickness=2, scale=2)

                        alert_position = get_next_alert_position(alerts)
                        add_alert(alerts, 'Knife Detected', alert_position)

    alerts = remove_expired_alerts(alerts)

    play_alert_sound(alerts)  # 알림 상태를 확인하고 소리 재생

    for alert_text, position, _, _ in alerts:
        cvzone.putTextRect(frame, alert_text, position, thickness=2, scale=2)

    cv2.imshow('Detection and Pose Estimation', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        draw_layer = not draw_layer

cap.release()
cv2.destroyAllWindows()
