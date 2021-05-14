import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('yuko.jpg')
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
