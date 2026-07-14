import numpy as np 
from ultralytics import YOLO
import cv2


cap = cv2.VideoCapture("tomato.mp4")
model = YOLO("AuTomodel/AuTomodel.pt")



while True:
    ret, frame= cap.read()
    
    # frame = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
    
    results = model(frame, conf = 0.5, iou = 0.6)
    result = results[0]
    
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype = int)
    classes = np.array(result.boxes.cls.cpu(), dtype = int)
    confi = np.array(result.boxes.conf.cpu(), dtype = float)
    names = model.names
    
    for bbox, cls, conf in zip(bboxes, classes, confi):
        (x,y,x2,y2) = bbox
        conf = round(conf, 4)
        cv2.rectangle(frame, (x,y), (x2,y2), [0,0,0], 3)
        cv2.putText(frame, names[cls], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, [255,255,255], 1 )
        cv2.putText(frame, str(conf), (x,y2), cv2.FONT_HERSHEY_COMPLEX, 1, [255,255,255], 1)
        
        if names[cls] == "Green (Unripe)":
             cv2.rectangle(frame, (100,0), (125,25), [0,255,0], 100)
        if names[cls] == "Breaking (Partially Unripe)":
             cv2.rectangle(frame, (200,25), (225,25), [150,200,225], 100)
        
        if names[cls] == "Turning (Paritally Ripe)":
             cv2.rectangle(frame, (300,25), (325,25), [0,128,255], 100)
             
        if names[cls] == "Bright Red (Ripe)":
             cv2.rectangle(frame, (400,25), (425,25), [0,69,255], 100)
             
        if names[cls] == "dark red (Over Ripe)":
             cv2.rectangle(frame, (500,25), (525,25), [0,0,255], 100)
             
        if names[cls] == "Damaged (Rejected)":
            cv2.rectangle(frame, (600,25), (625,25), [255,0,255], 100)
             
        
        

        
    
    

    cv2.imshow("video", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows()