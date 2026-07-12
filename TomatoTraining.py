from ultralytics import YOLO 
import numpy as np
import cv2

cap = cv2.VideoCapture("tomato.mp4")
model = YOLO("my_model/my_model.pt")


while True:
    ret, frame= cap.read()
    
    # frame = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
    
    results = model(frame)
    result = results[0]
    
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype = int)
    classes = np.array(result.boxes.cls.cpu(), dtype = int)
    confi = np.array(result.boxes.conf.cpu(), dtype = float)
    names = model.names
    
    for bbox, cls, conf in zip(bboxes, classes, confi):
        (x,y,x2,y2) = bbox
        conf = round(conf, 4)
        cv2.rectangle(frame, (x,y), (x2,y2), [0,0,0], 3)
        cv2.putText(frame, names[cls], (x,y), cv2.FONT_HERSHEY_COMPLEX, 2, [255,255,255], 2  )
        cv2.putText(frame, str(conf), (x,y2), cv2.FONT_HERSHEY_COMPLEX, 2, [255,255,255], 2 )
        
    
    

    cv2.imshow("video", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows()