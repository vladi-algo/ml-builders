import cv2
import numpy as np

ROI_top = 100
ROI_bottom = 900  # height 1000
ROI_right = 100
ROI_left = 700  # width 750

img = cv2.imread('test_gest_2.jpg')

img = img[50:ROI_bottom, 50:ROI_left]
print (img.shape)

gausBlur = cv2.GaussianBlur(img, (9,9),0)
cv2.imshow('Gaussian Blurring', gausBlur)
cv2.waitKey(0)

accumulated_weight=0.5
background = background.copy().astype("float")
print (type(background))
cv2.imshow('background', background)
cv2.waitKey(0)

cv2.accumulateWeighted(gausBlur, background, accumulated_weight)


cv2.destroyAllWindows()