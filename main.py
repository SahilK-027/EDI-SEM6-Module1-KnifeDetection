# =======================================================================================================
# Dependencies
# =======================================================================================================
from ultralytics import YOLO
import cv2
import cvzone
import math

# =======================================================================================================
# Creating webcam Object
# =======================================================================================================
cap = cv2.VideoCapture(0)
# Setting Width and height of webcam
cap.set(3, 1280)
cap.set(4, 720)

# video as feed
# cap = cv2.VideoCapture('URL')

# =======================================================================================================
# Creating YOLO model
# =======================================================================================================
model = YOLO('weights.pt')
classnames = ['knife']
limits1 = [0, 0, 1280, 0]
limits2 = [0, 720, 1280, 720]
limits3 = [0, 0, 0, 720]
limits4 = [1280, 0, 1280, 720]
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 8)
    cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 8)
    cv2.line(img, (limits3[0], limits3[1]), (limits3[2], limits3[3]), (0, 255, 0), 8)
    cv2.line(img, (limits4[0], limits4[1]), (limits4[2], limits4[3]), (0, 255, 0), 8)
    cvzone.putTextRect(img, f'Event : Normal Event', (50, 50))
    for i in results:
        boundingBoxes = i.boxes
        for box in boundingBoxes:
            # Drawing bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1

            # Drawing confidence value
            conf = math.ceil(box.conf[0] * 100) / 100
            cvzone.putTextRect(img, f'Event : Normal', (50, 50))
            # Drawing class name value
            cls = int(box.cls[0])
            currentClass = classnames[cls]
            # if currentClass == 'knife' and conf > 0.3:
            cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(40, y1)))
            cvzone.cornerRect(img, (x1, y1, w, h))
            cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 8)
            cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 8)
            cv2.line(img, (limits3[0], limits3[1]), (limits3[2], limits3[3]), (0, 0, 255), 8)
            cv2.line(img, (limits4[0], limits4[1]), (limits4[2], limits4[3]), (0, 0, 255), 8)
            cvzone.putTextRect(img, f'Event : ANOMALY DETECTED', (50, 50))

    cv2.imshow('Image', img)
    cv2.waitKey(1)