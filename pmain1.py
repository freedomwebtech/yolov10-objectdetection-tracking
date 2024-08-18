import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
from tracker import*
import numpy as np

model = YOLO("yolov10n.pt")  
 
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


cap=cv2.VideoCapture('road.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

tracker1=Tracker()
tracker2=Tracker()
going_up_truck={}
truck_up=[]
going_down_truck={}
truck_down=[]

going_up_car={}
car_up=[]
going_down_car={}
car_down=[]
count=0
cy1=287
cy2=305
offset=6
while True:
    ret,frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
    frame = cv2.resize(frame, (1020, 600))

    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list1=[]
    truck=[]
    list2=[]
    car=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        
        if 'car' in c:
           list1.append([x1,y1,x2,y2]) 
           car.append(c)
        if 'truck' in c:
           list2.append([x1,y1,x2,y2]) 
           truck.append(c)   
    
            
    bbox_idx1=tracker1.update(list1)
    bbox_idx2=tracker2.update(list2)

    for bbox1 in bbox_idx1:
        x3,y3,x4,y4,id=bbox1
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if 'car' in car:
           if cy1<(cy+offset) and cy1>(cy-offset):
              going_up_car[id]=(cx,cy)
           if id in going_up_car:
              if cy2<(cy+offset) and cy2>(cy-offset): 
                 cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
                 cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
                 cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                 if car_up.count(id)==0:
                    car_up.append(id)
           if cy2<(cy+offset) and cy2>(cy-offset):
              going_down_car[id]=(cx,cy)
           if id in going_down_car:
              if cy1<(cy+offset) and cy1>(cy-offset): 
                 cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
                 cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
                 cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,0),2)
                 if car_down.count(id)==0:
                    car_down.append(id)
               

########################################################################                 
    for bbox2 in bbox_idx2:
        x5,y5,x6,y6,id1=bbox2
        cx3=int(x5+x6)//2
        cy3=int(y5+y6)//2
        if 'truck' in truck:
           if cy1<(cy3+offset) and cy1>(cy3-offset):
              going_up_truck[id1]=(cx3,cy3)
           if id1 in going_up_truck:
              if cy2<(cy3+offset) and cy2>(cy3-offset): 
                 cv2.circle(frame,(cx3,cy3),4,(255,0,0),-1)
                 cvzone.putTextRect(frame,f'{id}',(x5,y5),1,1)
                 cv2.rectangle(frame,(x5,y5),(x6,y6),(0,255,0),2)
                 if truck_up.count(id1)==0:
                    truck_up.append(id1)
           if cy2<(cy3+offset) and cy2>(cy3-offset):
              going_down_truck[id1]=(cx3,cy3)
           if id1 in going_down_truck:
              if cy1<(cy3+offset) and cy1>(cy3-offset):
                 cv2.circle(frame,(cx3,cy3),4,(255,255,0),-1)
                 cvzone.putTextRect(frame,f'{id1}',(x5,y5),1,1)
                 cv2.rectangle(frame,(x5,y5),(x6,y6),(255,255,0),2)
                 if truck_down.count(id1)==0: 
                    truck_down.append(id1)
                     


         

                      
                 
                 
    print(len(truck_up))
    print(len(truck_down))
    print(len(car_up))
    print(len(car_down))
    cv2.line(frame,(203,287),(1019,287),(255,0,255),2)

    cv2.line(frame,(165,305),(1019,305),(255,255,255),2)
    cv2.imshow("RGB", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


