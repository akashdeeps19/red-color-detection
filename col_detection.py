import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def preprocessing(img_org):
  img = cv2.resize(img_org,(300,300),cv2.INTER_AREA)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower_blue = np.array([0,50,50])
  upper_blue = np.array([10,255,255])
  mask1 = cv2.inRange(hsv, np.array([0, 120, 110]), np.array([8, 255, 255]))
  mask2 =  cv2.inRange(hsv,np.array([172, 120, 110]), np.array([180, 255, 255]))
  mask = mask1+mask2  
  res = cv2.bitwise_and(img,img, mask= mask)
  im_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
  (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  im_bw = cv2.medianBlur(im_bw,3)
  
  ret, labels = cv2.connectedComponents(im_bw)
  regions_x = [[] for i in range(ret)]
  regions_y = [[] for i in range(ret)]
  img_res = img
  for i in range(300):
    for j in range(300):
      if(labels[i,j] > 0):
          regions_x[labels[i,j]].append(i)
          regions_y[labels[i,j]].append(j)
  for i in range(1,ret):
    s_point = (min(regions_y[i]),min(regions_x[i]))
    e_point = (max(regions_y[i]),max(regions_x[i]))
    img_res = cv2.rectangle(img,s_point,e_point,(255,0,0),1)
  return img_res

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    frame = preprocessing(frame)
    frame = cv2.resize(frame,(1000,600),cv2.INTER_AREA)
    cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else: 
    break

cap.release()
cv2.destroyAllWindows()