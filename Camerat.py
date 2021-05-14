import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    #frame = cv2.Canny(frame, 50, 125, L2gradient=True)
    cv2.imshow('frame', frame)
    c = cv2.waitKey(1)
    if c==27 or c == ord('q'): #esc key
        break
    
cap.release()
cv2.destroyAllWindows()