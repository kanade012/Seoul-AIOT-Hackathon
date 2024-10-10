import cv2
import cvzone
import torch
import numpy as np
import math
from ultralytics import YOLO
import time

# Load YOLOv8 models: one for object detection (knife) and one for pose estimation (fall detection)
knife_model = YOLO('best.pt')  # Custom trained object detection model (e.g., for knife detection)
pose_model = YOLO('yolov8n-pose.pt')  # YOLOv8 pose estimation model

# Initialize video capture (webcam or video file)
cap = cv2.VideoCapture("How People Walk.mp4")

# Define keypoints for pose estimation
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Person tracking for fall detection
person_data = {}
fall_duration_threshold = 0.8  # Time threshold to detect a fall
position_reset_threshold = 2.0  # Time threshold to consider a fall if posture doesn't return

# Initialize knife detection time tracking
knife_detection_times = {}  # Dictionary to track knife detection time for each knife

# List to store active alerts and their positions (now includes a timestamp for managing alert expiry)
alerts = []
alert_spacing = 60  # Space between alerts to avoid overlap
alert_expiry_time = 5  # Time after which an alert expires (optional)

# Function to detect fall based on keypoints
def detect_fall(initial_positions, current_keypoints):
    try:
        neck = current_keypoints[5]  # Neck keypoint
        hip = current_keypoints[11]  # Hip keypoint (left hip)
        initial_neck = initial_positions[5]
        initial_hip = initial_positions[11]

        # Check if both the neck and hip have moved significantly down
        if neck[1] > initial_neck[1] + 100 and hip[1] > initial_hip[1] + 100:
            return True
    except (IndexError, TypeError) as e:
        print(f"Error in fall detection: {e}")
        return False
    return False

# Function to assign a unique position for alerts without overlap
def get_next_alert_position(alerts):
    y_base = 50
    return (50, y_base + len(alerts) * alert_spacing)

# Function to add an alert only if it does not exist already
def add_alert(alerts, alert_text, position):
    for alert, _, _ in alerts:
        if alert == alert_text:  # Check if the alert text already exists
            return
    # If alert is not a duplicate, add it with the current time
    alerts.append((alert_text, position, time.time()))

# Function to remove expired alerts (optional, based on expiry time)
def remove_expired_alerts(alerts):
    current_time = time.time()
    return [(text, pos, t) for text, pos, t in alerts if current_time - t < alert_expiry_time]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (980, 740))

    # Run object detection (knife detection) model
    knife_results = knife_model(frame)

    # Run pose estimation (fall detection) model
    with torch.no_grad():
        pose_results = pose_model(frame)

    # Process knife detection results
    for knife_info in knife_results:
        knife_boxes = knife_info.boxes
        for box in knife_boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            conf = math.ceil(confidence * 100)

            # If a knife is detected and confidence is above the threshold
            if conf > 50 and class_detect == 0:  # Assuming class 0 is 'knife'
                width = x2 - x1
                height = y2 - y1
                knife_area = width * height

                # Track the knife detection time
                knife_id = (x1, y1, x2, y2)  # Unique knife identifier based on its coordinates
                if knife_id not in knife_detection_times:
                    knife_detection_times[knife_id] = {'start_time': time.time(), 'valid': False}

                # Check if the knife is smaller than the person
                for person_id, pose_result in enumerate(pose_results):
                    keypoints = pose_result.keypoints.xy[0]

                    # If no keypoints, skip this person
                    if len(keypoints) < 6:  # We need at least 6 keypoints for comparison
                        continue

                    person_height = int(abs(keypoints[5][1] - keypoints[15][1]))  # From left shoulder to left ankle
                    person_width = int(abs(keypoints[6][0] - keypoints[12][0]))   # From right shoulder to right hip
                    person_area = person_height * person_width

                    if knife_area < person_area:  # Ensure knife is smaller than the person
                        detection_time = time.time() - knife_detection_times[knife_id]['start_time']

                        # Check if the knife has been detected for more than 1 second
                        if detection_time >= 1.0:
                            knife_detection_times[knife_id]['valid'] = True  # Mark as valid detection
                            if knife_detection_times[knife_id]['valid']:
                                # Draw the knife bounding box and add alert if detection is valid
                                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                                cvzone.putTextRect(frame, "Knife", [x1 + 8, y1 - 12], thickness=2, scale=2)

                                # Add knife detection alert (if not already present)
                                alert_position = get_next_alert_position(alerts)
                                add_alert(alerts, f'Knife Detected: Person {person_id}', alert_position)

    # Process pose estimation results for fall detection
    if pose_results and pose_results[0].keypoints is not None:
        for person_id, pose_result in enumerate(pose_results):
            keypoints = pose_result.keypoints.xy[0]  # Extract keypoints for each person

            # Check if keypoints have the necessary length (e.g., at least 6 keypoints)
            if len(keypoints) < 6:  # We need at least 6 keypoints to use the left shoulder (index 5)
                continue  # Skip this person if keypoints are incomplete

            # Unique person tracking using left shoulder keypoint (index 5) for example
            person_key = tuple(keypoints[5].tolist())  # Use left shoulder coordinates as a unique key
            if person_key not in person_data:
                person_data[person_key] = {
                    'initial_positions': keypoints.clone(),  # Store initial positions of keypoints
                    'fall_start_time': None
                }

            # Draw keypoints on the frame
            for i, kp in enumerate(keypoints):
                if i < len(keypoint_names):
                    x, y = int(kp[0].item()), int(kp[1].item())
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, keypoint_names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Fall detection for each person
            initial_positions = person_data[person_key]['initial_positions']
            if detect_fall(initial_positions, keypoints):
                if person_data[person_key]['fall_start_time'] is None:
                    person_data[person_key]['fall_start_time'] = time.time()
                else:
                    fall_duration = time.time() - person_data[person_key]['fall_start_time']
                    if fall_duration >= fall_duration_threshold:
                        # Add fall detection alert (if not already present)
                        alert_position = get_next_alert_position(alerts)
                        add_alert(alerts, f'Fall Detected: Person {person_id}', alert_position)

    # Remove expired alerts if the expiry mechanism is used
    alerts = remove_expired_alerts(alerts)

    # Display all alerts without overlap
    for alert_text, position, _ in alerts:
        cvzone.putTextRect(frame, alert_text, position, thickness=2, scale=2)

    # Show the combined results
    cv2.imshow('Detection and Pose Estimation', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
