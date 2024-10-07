import cv2
import cvzone
import torch
import numpy as np
from ultralytics import YOLO
import time  # 시간 추적을 위해 추가

# YOLOv8 포즈 추정 모델 로드 (사람의 관절을 추정하는 모델)
model = YOLO('yolov8n-pose.pt')  # 포즈 추정용 모델

# GPU 또는 CPU 사용 설정 (YOLO에서는 자동 처리)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cap = cv2.VideoCapture("fall.mp4")

# 관절 이름 리스트
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# 낙상 감지 시간 추적을 위한 변수
fall_start_time = None  # 낙상 감지 시작 시간
fall_duration_threshold = 0.8  # 0.8초 동안 지속되면 낙상으로 인식

def detect_fall(keypoints):
    try:
        head = keypoints[0]  # 머리 좌표
        neck = keypoints[5]  # 목 좌표 (오른쪽 어깨와 왼쪽 어깨 중간을 목으로 대체)
        hip = keypoints[11]  # 왼쪽 힙 좌표 (허리 좌표 대체)
        knee = keypoints[13]  # 왼쪽 무릎 좌표

        # 기준을 낮추어 낙상을 더 민감하게 감지
        # 허리(hip)가 무릎보다 조금만 아래에 있고, 목이 허리보다 조금 위 또는 비슷한 높이에 있을 때도 낙상으로 간주
        if hip[1] > (knee[1] - 50) and neck[1] > (hip[1] - 30):
            return True
    except (IndexError, TypeError) as e:
        print(f"Error in fall detection: {e}")
        return False

    return False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 이미지 크기를 맞추고 전처리
    frame = cv2.resize(frame, (980, 740))

    # 추론 과정 (torch.no_grad로 그래디언트 계산 비활성화)
    with torch.no_grad():
        results = model(frame)  # 모델 추론 수행

    fall_detected = False

    # 포즈 결과 처리 (감지된 객체에서 keypoints 추출)
    if results and results[0].keypoints is not None:  # 키포인트가 있을 때만 처리
        for result in results:
            keypoints = result.keypoints.xy  # 관절 좌표 추출 (shape: [num_people, num_keypoints, 2])

            for person in keypoints:
                # 관절 좌표 그리기 및 이름 표시
                for i, kp in enumerate(person):
                    if i < len(keypoint_names):  # 인덱스가 keypoint_names 리스트 길이보다 작을 때만 실행
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 각 관절에 원 그리기
                        # 해당 관절 이름을 표시
                        cv2.putText(frame, keypoint_names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # 낙상 감지
                if detect_fall(person):
                    fall_detected = True

    # 낙상 감지가 되었을 경우 타이머 시작
    if fall_detected:
        if fall_start_time is None:  # 처음 감지될 경우 타이머 시작
            fall_start_time = time.time()
        else:
            # 타이머가 시작된 이후 경과 시간 확인
            fall_duration = time.time() - fall_start_time
            if fall_duration >= fall_duration_threshold:  # 0.8초 이상 감지되면 낙상으로 인식
                cvzone.putTextRect(frame, 'Fall Detected', [50, 50], thickness=2, scale=3)
    else:
        # 낙상 감지가 되지 않으면 타이머 초기화
        fall_start_time = None

    # 프레임 보여주기
    cv2.imshow('Pose Detection', frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
