import cv2
import cvzone
import torch
import numpy as np
import math
from ultralytics import YOLO
import time

# Load YOLOv8 models: one for human detection, one for object detection (knife), and one for pose estimation (fall detection)
person_model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8n model for person detection
knife_model = YOLO('best.pt')  # Custom trained object detection model (e.g., for knife detection)
pose_model = YOLO('yolov8n-pose.pt')  # YOLOv8 pose estimation model

# Initialize video capture (webcam or video file)
cap = cv2.VideoCapture("fall.mp4")

# Define keypoints for pose estimation
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Person tracking for fall detection
person_data = {}  # To store person ID, their last known position, and fall start time
person_id_counter = 0  # To assign a unique ID to each person
fall_duration_threshold = 1.0  # Time threshold to detect a fall (1 second)

# Initialize knife detection time tracking
knife_detection_times = {}  # Dictionary to track knife detection time for each knife

# List to store active alerts and their positions (now includes a timestamp for managing alert expiry)
alerts = []
alert_spacing = 60  # Space between alerts to avoid overlap
alert_expiry_time = 5  # Time after which an alert expires (optional)

# Variable to control whether to draw the detection layers
draw_layer = True  # By default, layers are drawn

# Function to assign a unique ID to each person
def get_person_id(person_boxes, person_data, threshold=50):
    global person_id_counter

    assigned_ids = []
    for box in person_boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        current_person_pos = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Person's center position

        # Compare with existing persons to see if they are the same
        matched_id = None
        for person_id, last_position in person_data.items():
            if 'position' in last_position:
                dist = np.linalg.norm(current_person_pos - last_position['position'])  # Euclidean distance
                if dist < threshold:
                    matched_id = person_id
                    person_data[person_id]['position'] = current_person_pos  # Update position
                    break

        # If no match, assign a new ID
        if matched_id is None:
            person_id_counter += 1
            person_data[person_id_counter] = {'position': current_person_pos}
            matched_id = person_id_counter

        assigned_ids.append(matched_id)

    return assigned_ids

# Function to detect fall and track fall duration
def detect_fall_with_duration(person_id, current_keypoints, person_data):
    try:
        neck = current_keypoints[5]  # Neck keypoint
        hip = current_keypoints[11]  # Left Hip keypoint
        left_ankle = current_keypoints[15]  # Left Ankle keypoint
        right_ankle = current_keypoints[16]  # Right Ankle keypoint

        # Calculate the Y-axis differences between neck, hip, and ankles
        neck_hip_diff = abs(neck[1] - hip[1])  # Difference between neck and hip Y position
        hip_ankle_diff = abs(hip[1] - min(left_ankle[1], right_ankle[1]))  # Difference between hip and ankles Y position

        # Thresholds for detecting fall: small differences in Y positions indicate a fallen posture
        neck_hip_threshold = 100  # Neck and hip should be close together (fallen posture)
        hip_ankle_threshold = 100  # Hip and ankles should also be close together (legs on the ground)

        # Check if both neck-hip and hip-ankle Y differences are below the thresholds (potential fall)
        if neck_hip_diff < neck_hip_threshold and hip_ankle_diff < hip_ankle_threshold:
            # Check if this person is already in fall detection mode
            if 'fall_start_time' not in person_data[person_id]:
                person_data[person_id]['fall_start_time'] = time.time()  # Start the fall timer
            else:
                # Check how long the fall has been detected
                fall_duration = time.time() - person_data[person_id]['fall_start_time']
                if fall_duration >= fall_duration_threshold:
                    return True  # Fall detected for longer than threshold
        else:
            # Reset fall detection if the person recovers
            if 'fall_start_time' in person_data[person_id]:
                del person_data[person_id]['fall_start_time']  # Reset fall tracking if posture is recovered
    except (IndexError, TypeError) as e:
        print(f"Error in fall detection: {e}")
        return False

    return False

# Function to assign a unique position for alerts without overlap
def get_next_alert_position(alerts):
    y_base = 50
    return (50, y_base + len(alerts) * alert_spacing)

# Function to add an alert only if it does not exist already (check both text and position)
def add_alert(alerts, alert_text, position, person_id):
    # Check if the same alert text and person ID already exist in the alerts list
    for existing_alert, existing_pos, existing_id, _ in alerts:
        if existing_alert == alert_text and existing_id == person_id:
            return  # If the same alert for the same person already exists, don't add it
    # If alert is not a duplicate, add it with the current time
    alerts.append((alert_text, position, person_id, time.time()))

# Function to remove expired alerts (optional, based on expiry time)
def remove_expired_alerts(alerts):
    current_time = time.time()
    return [(text, pos, obj_id, t) for text, pos, obj_id, t in alerts if current_time - t < alert_expiry_time]

# Main loop for processing frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame if needed
    frame = cv2.resize(frame, (980, 740))

    # Run YOLOv8n person detection model
    person_results = person_model(frame)

    # Process person detection results
    for person_info in person_results:
        person_boxes = person_info.boxes
        person_ids = get_person_id(person_boxes, person_data)  # Get unique person IDs

        for person_box, person_id in zip(person_boxes, person_ids):
            x1, y1, x2, y2 = person_box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = person_box.conf[0]
            class_detect = person_box.cls[0]
            class_detect = int(class_detect)
            conf = math.ceil(confidence * 100)

            # If a person is detected and confidence is above the threshold
            if conf > 50 and class_detect == 0:  # Assuming class 0 is 'person'
                width = x2 - x1
                height = y2 - y1

                # Draw the person bounding box if drawing layers is enabled
                if draw_layer:
                    cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                    cvzone.putTextRect(frame, f"Person {person_id}", [x1 + 8, y1 - 12], thickness=2, scale=2)

                # Crop the frame around the detected person
                person_roi = frame[y1:y2, x1:x2]

                # Run knife detection (best.pt) on the person
                knife_results = knife_model(person_roi)

                # Run pose estimation (yolov8n-pose) on the person
                with torch.no_grad():
                    pose_results = pose_model(person_roi)

                # Process knife detection results
                for knife_info in knife_results:
                    knife_boxes = knife_info.boxes
                    for box in knife_boxes:
                        x1_k, y1_k, x2_k, y2_k = box.xyxy[0]
                        x1_k, y1_k, x2_k, y2_k = int(x1_k), int(y1_k), int(x2_k), int(y2_k)
                        confidence_k = box.conf[0]
                        class_detect_k = box.cls[0]
                        class_detect_k = int(class_detect_k)
                        conf_k = math.ceil(confidence_k * 100)

                        # If a knife is detected and confidence is above the threshold
                        if conf_k > 50 and class_detect_k == 0:  # Assuming class 0 is 'knife'
                            width_k = x2_k - x1_k
                            height_k = y2_k - y1_k
                            knife_area = width_k * height_k

                            # Track the knife detection time
                            knife_id = (x1_k, y1_k, x2_k, y2_k)  # Unique knife identifier based on its coordinates
                            if knife_id not in knife_detection_times:
                                knife_detection_times[knife_id] = {'start_time': time.time(), 'valid': False}

                            # Ensure knife is smaller than the person
                            person_height = height
                            person_width = width
                            person_area = person_height * person_width

                            if knife_area < person_area:  # Ensure knife is smaller than the person
                                detection_time = time.time() - knife_detection_times[knife_id]['start_time']

                                # Check if the knife has been detected for more than 1 second
                                if detection_time >= 1.0:
                                    knife_detection_times[knife_id]['valid'] = True  # Mark as valid detection
                                    if knife_detection_times[knife_id]['valid']:
                                        # Draw the knife bounding box if drawing layers is enabled
                                        if draw_layer:
                                            cvzone.cornerRect(frame, [x1_k + x1, y1_k + y1, width_k, height_k], l=30, rt=6)
                                            cvzone.putTextRect(frame, "Knife", [x1_k + 8 + x1, y1_k - 12 + y1], thickness=2, scale=2)

                                        # Add knife detection alert (if not already present)
                                        alert_position = get_next_alert_position(alerts)
                                        add_alert(alerts, f'Knife Detected: Person {person_id}', alert_position, knife_id)

                # Process pose estimation results for fall detection
                if pose_results and pose_results[0].keypoints is not None:
                    for pose_result in pose_results:
                        keypoints = pose_result.keypoints.xy[0]

                        # Check if keypoints have the necessary length
                        if len(keypoints) < 6:
                            continue

                        # Draw keypoints on the frame if drawing layers is enabled
                        if draw_layer:
                            for i, kp in enumerate(keypoints):
                                if i < len(keypoint_names):
                                    x, y = int(kp[0].item()) + x1, int(kp[1].item()) + y1
                                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                                    cv2.putText(frame, keypoint_names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 0, 0), 1)

                        # Fall detection for each person (only alert if fall lasts more than 1 second)
                        if detect_fall_with_duration(person_id, keypoints, person_data):
                            alert_position = get_next_alert_position(alerts)
                            add_alert(alerts, f'Fall Detected: Person {person_id}', alert_position, person_id)

    # Remove expired alerts
    alerts = remove_expired_alerts(alerts)

    # Display all alerts without overlap (always visible regardless of layer drawing)
    for alert_text, position, _, _ in alerts:
        cvzone.putTextRect(frame, alert_text, position, thickness=2, scale=2)

    # Show the results
    cv2.imshow('Detection and Pose Estimation', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        draw_layer = not draw_layer  # Toggle drawing layers on/off with 'w'

cap.release()
cv2.destroyAllWindows()
