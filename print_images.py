import copy
import cv2
import numpy as np
from keras.models import load_model
import time

# General Settings
prediction = ''
action = ''
score = 0
img_counter = 500


# Turn on/off the ability to save images, or control Philips Hue/Sonos
save_images, selected_gesture = False, 'peace'

gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}



# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variableslt
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works


def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)

#Capture background
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
print('Background captured')

while camera.isOpened():

    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    cv2.imshow('original', frame)

    # Run once background is captured
    
    img = remove_background(frame)
    img = img[0:int(cap_region_y_end * frame.shape[0]),
         int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
    # cv2.imshow('mask', img)

    # convert the image into binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    # cv2.imshow('blur', blur)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Add prediction and action text to thresholded image
    # cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    # cv2.putText(thresh, f"Action: {action}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))  # Draw the text
    # Draw the text
    #cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #            (255, 255, 255))
    #cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #            (255, 255, 255))  # Draw the text
    cv2.imshow('ori', thresh)
    '''
    # get the contours
    thresh1 = copy.deepcopy(thresh)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        cv2.imshow('output', drawing)
    '''
    # Keyboard OP
    k = cv2.waitKey(10)

    if k == 27:  # press ESC to exit all windows at any time
        break

    elif k == 32: #space

            print ("image printing...") 
            img_name = f"./printed_images/image_id_replace_by_param_{img_counter}.jpg".format(
                img_counter)
            #cv2.imwrite(img_name, thresh)
            cv2.imwrite(img_name, drawing)
            print("{} written".format(img_name))


            img_counter += 1

    elif k == ord('t'):

        print('Tracker turned on.')

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        # Select Region of Interest (ROI)
        r = cv2.selectROI(frame)

        # Crop image
        imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # setup initial location of window
        r, h, c, w = 250, 400, 400, 400
        track_window = (c, r, w, h)
        # set up the ROI for tracking
        roi = imCrop
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        while (1):
            ret, frame = cap.read()
            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                # apply meanshift to get the new location
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                # Draw it on image
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cv2.imshow('img2', img2)
                k = cv2.waitKey(60) & 0xff
                if k == 27:  # if ESC key
                    break
                else:
                    cv2.imwrite(chr(k) + ".jpg", img2)
            else:
                break
        cv2.destroyAllWindows()
        cap.release()
