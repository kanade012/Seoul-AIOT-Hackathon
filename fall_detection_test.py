import cv2
import cvzone
import torch
import numpy as np
from ultralytics import YOLO
import time

# YOLOv8 포즈 추정 모델 로드 (사람의 관절을 추정하는 모델)
model = YOLO('model/yolov8n-pose.pt')  # 포즈 추정용 모델

# GPU 또는 CPU 사용 설정 (YOLO에서는 자동 처리)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cap = cv2.VideoCapture(0)

# 관절 이름 리스트
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# 사람별 상태 추적을 위한 딕셔너리
person_data = {}

# 낙상 감지 시간 및 기준
fall_duration_threshold = 0.8  # 0.8초 동안 지속되면 낙상으로 인식
position_reset_threshold = 2.0  # 목과 척추가 몇 초 내로 정상 위치로 돌아오지 않으면 낙상으로 간주

def detect_fall(initial_positions, current_keypoints):
    try:
        neck = current_keypoints[5]  # 목 좌표
        hip = current_keypoints[11]  # 왼쪽 힙 좌표 (허리 좌표 대체)

        initial_neck = initial_positions[5]
        initial_hip = initial_positions[11]

        # 현재 목과 척추가 초기 위치에 비해 크게 아래로 이동했는지 확인
        if neck[1] > initial_neck[1] + 100 and hip[1] > initial_hip[1] + 100:
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
        for person_id, result in enumerate(results):  # 각 사람에 대해 반복
            keypoints = result.keypoints.xy[0]  # 관절 좌표 추출 (여러 명 중 첫 번째 사람의 keypoints 사용)

            if person_id not in person_data:
                # 처음 본 사람일 경우, 초기 위치와 타이머 추가 (clone()으로 복사)
                person_data[person_id] = {
                    'initial_positions': keypoints.clone(),  # clone()을 사용하여 텐서 복사
                    'fall_start_time': None
                }

            # 관절 좌표 그리기 및 이름 표시
            for i, kp in enumerate(keypoints):
                if i < len(keypoint_names):  # 인덱스가 keypoint_names 리스트 길이보다 작을 때만 실행
                    x, y = int(kp[0].item()), int(kp[1].item())  # 각 키포인트 좌표에서 x, y 추출
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 각 관절에 원 그리기
                    # 해당 관절 이름을 표시
                    cv2.putText(frame, keypoint_names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 낙상 감지
            initial_positions = person_data[person_id]['initial_positions']
            if detect_fall(initial_positions, keypoints):
                if person_data[person_id]['fall_start_time'] is None:
                    # 처음 낙상이 감지된 경우 타이머 시작
                    person_data[person_id]['fall_start_time'] = time.time()
                else:
                    # 낙상 지속 시간 확인
                    fall_duration = time.time() - person_data[person_id]['fall_start_time']
                    if fall_duration >= fall_duration_threshold:
                        cvzone.putTextRect(frame, f'Fall Detected: Person {person_id}', [50, 50 + person_id * 50], thickness=2, scale=3)
            else:
                # 낙상이 감지되지 않으면 타이머 초기화
                person_data[person_id]['fall_start_time'] = None

    # 프레임 보여주기
    cv2.imshow('Pose Detection', frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
