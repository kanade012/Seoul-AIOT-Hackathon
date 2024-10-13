import cv2
import cvzone
import math
from ultralytics import YOLO

model = YOLO('model/knife_model.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (980, 740))

    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            conf = math.ceil(confidence * 100)

            print(f"Class ID: {class_detect}, Confidence: {conf}%")

          
            if conf > 50 and class_detect == 0:  
                width = x2 - x1
                height = y2 - y1
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, "Knife", [x1 + 8, y1 - 12], thickness=2, scale=2)

            else:
                pass

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()
