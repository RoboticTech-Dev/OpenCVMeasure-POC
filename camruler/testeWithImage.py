import numpy as np
import cv2

path = "camruler/TestImage.png"
img = cv2.imread(path)



low_color  = np.array([0, 155, 84])
high_color = np.array([179, 255, 255])

def setHSV(Val):
    print(Val)

img1 = np.zeros((300,512,3),np.uint8)
cv2.namedWindow('LOW')

img2 = np.zeros((300,512,3),np.uint8)
cv2.namedWindow('HIGH')


cv2.createTrackbar('H','LOW',0,255,setHSV)
cv2.createTrackbar('S','LOW',0,255,setHSV)
cv2.createTrackbar('V','LOW',0,255,setHSV)

cv2.createTrackbar('H','HIGH',0,255,setHSV)
cv2.createTrackbar('S','HIGH',0,255,setHSV)
cv2.createTrackbar('V','HIGH',0,255,setHSV)

while True:

    color_mask = cv2.inRange(img,low_color,high_color)
    imgHSV     = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    cv2.imshow('LOW',img1)
    cv2.imshow('HIGH',img2)
    # cv2.imshow('imageHSV',imgHSV)
    cv2.imshow('image Mask',color_mask)

    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break

    H_l = cv2.getTrackbarPos('H','LOW')
    S_l = cv2.getTrackbarPos('S','LOW')
    V_l = cv2.getTrackbarPos('V','LOW')

    H_h = cv2.getTrackbarPos('H','HIGH')
    S_h = cv2.getTrackbarPos('S','HIGH')
    V_h = cv2.getTrackbarPos('V','HIGH')

    img1[:] = [H_l,S_l,V_l]
    img2[:] = [H_h,S_h,V_h]

    low_color  = np.array([H_l,S_l,V_l])
    high_color  = np.array([H_h,S_h,V_h])
 
cv2.destroyAllWindows()
    
    


