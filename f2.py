import cv2
from ultralystics import YOLO
from imultis.video  import WebcamVideoStream
import time 

model= YOLO("yolov8m.pt")
cap=WebcamVideoStream(src=0).start()
time.sleep(2)

while True:
  image=cap.read()
  #print (image.shape[:2]
  if image is None: continue

  results=model.predict(image)
  result= results[0]

for box in result.boxes:#rettangoli fuori dagli oggetti 
  class_id= result.names[box.cls[0].item()]
  cords= box.xyxy[0].tolist()
  cords=[round(x) for x in cords]
  conf=round(box.conf[0].item(),2)

  centroid=((cords[2]-cords[0])/2,(cords[3]-cords[1])/2) #cooordinata x=2  naoglo alto a sinistra / 2 e trovo la metà della x , fatto ciò per y e trovo centro 
  centroid=[round(x) for x in centroid]

  print("Object type:",class_id)
  print("Coordinates:", cords)
  print("Centroid:",centroid)
  print("Probabbility:", conf)
  print("------------------")

  t_cords=tuple(cords)
  cv2.rectangle(image,(t_cords[0],t_cords[1]),(t_cords[2],t_cords[3]),(0,255,0),thickness=3)# 23,230,210  giallo
  cv2.putText(image,class_id,(cords[0]+55,cords[1]-10),
  cv2.FONT_HERSHEY_SIMPLEX,1.0(0,0,255),2)
  centro=[t_cords[0]+centroid[0],t_cords[1]+centroid[1]]

  cv2.nameWindow('Camera',cv2.WINDOW_NORMAL)
  cv2.imShow("Camera",image)
  if cv2.waitKey(2) & 0xFF == ord('q'):break

cv2.destroyAllWindows()
cap.stop()



  