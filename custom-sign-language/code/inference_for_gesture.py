from utils.definitions import word_dict,PREDICTION_THRESHOLD
import numpy as np
import cv2
#import keras
from tensorflow.keras import models
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import boto3
import os

model = models.load_model(r"../model/best_model_custom_gestures.h5")

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 1000 #height
ROI_right = 100
ROI_left = 750 #width

def text_to_polly_sound(input_text):
    polly_client = boto3.client('polly')
    
    response = polly_client.synthesize_speech(VoiceId='Matthew',
                                            Engine='neural',
                                            OutputFormat='mp3', 
                                            Text = input_text)

    file = open('speech.mp3', 'wb')
    file.write(response['AudioStream'].read())
    file.close()

    dir_path = os.path.dirname(os.path.realpath(__file__)).replace(" ", "\ ") + '/speech.mp3'

    os.system(("afplay " + dir_path))

def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)



def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    #Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)

cam = cv2.VideoCapture(0)
num_frames =0
while True:
    ret, frame = cam.read()

    # filpping the frame to prevent inverted image of captured frame...
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # ROI from the frame
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)


    if num_frames < 70:
        
        cal_accum_avg(gray_frame, accumulated_weight)
        
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT!", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    else: 
        # segmenting the hand region
        hand = segment_hand(gray_frame)

        # Checking if we are able to detect the hand...
        if hand is not None:
            
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)

            #(white / black)
            named_window = "Thesholded Hand Image-B/W"
            cv2.namedWindow(named_window)
            cv2.moveWindow(named_window, 850,100)
            cv2.imshow(named_window, thresholded)
            
            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
            
            pred = model.predict(thresholded)
            # print all predictions per gesture
            print(pred)
            #x_axis = pred[0]
            #if len(x_axis) > 0 and x_axis[np.argmax(x_axis)] >= PREDICTION_THRESHOLD:
            cv2.putText(frame_copy, "PREDICTED gesture:" + word_dict[np.argmax(pred)], (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)

    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand

    cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    cv2.imshow("Sign Detection", frame_copy)


    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()
