from utils.definitions import word_dict, PREDICTION_THRESHOLD
import numpy as np
import cv2
# import keras
from tensorflow.keras import models
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import boto3
import os
import time
import pygame
import pygame._sdl2 as sdl2
from pygame.locals import *

model = models.load_model(r"../model/best_model_custom_gestures.h5")
polly_client = boto3.client('polly')
# pygame.mixer.pre_init(devicename="BlackHole 16ch")
# pygame.mixer.pre_init(devicename="BlackHole 16ch")
pygame.mixer.init()

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 900  # height 1000
ROI_right = 100
ROI_left = 700  # width 750

#Joanna
def text_to_polly_sound(input_text):
    response = polly_client.synthesize_speech(VoiceId='Matthew',
                                              Engine='neural',
                                              OutputFormat='mp3',
                                              Text=input_text)
    file = open('speech.mp3', 'wb')
    file.write(response['AudioStream'].read())
    file.close()
    pygame.mixer.music.load("speech.mp3")
    pygame.mixer.music.play()
    # time.sleep(1)


def cal_accum_avg(pframe, accumulated_weight):
    global background

    if background is None:
        background = pframe.copy().astype("float")
        return None

    cv2.accumulateWeighted(pframe, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
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
num_frames = 0
predicted_gest = ""
is_voiceless = False
pygame.init()

while True:
    ret, frame = cam.read()

    # filpping the frame to prevent inverted image of captured frame...
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # ROI from the frame
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 200:

        cal_accum_avg(gray_frame, accumulated_weight)

        cv2.putText(frame_copy, "FETCHING BACKGROUND. PLEASE WAIT...", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2)
    else:
        # segmenting the hand region
        hand = segment_hand(gray_frame)

        # Checking if we are able to detect the hand...
        if hand is not None:
            is_voiceless = False
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

            # (white / black)
            named_window = "Threshold Hand Image-B/W"
            cv2.namedWindow(named_window)
            cv2.moveWindow(named_window, 850, 120)
            cv2.imshow(named_window, thresholded)

            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded, (1, thresholded.shape[0], thresholded.shape[1], 3))

            pred = model.predict(thresholded)
            # print all predictions per gesture
            # print(pred)
            x_axis = pred[0]
            if len(x_axis) > 0 and x_axis[np.argmax(x_axis)] >= PREDICTION_THRESHOLD:
                cv2.putText(frame_copy, "Recognized => " + word_dict[np.argmax(pred)], (60, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 64, 0), 2)
                predicted_gest = word_dict[np.argmax(pred)]
                # k = cv2.waitKey(50)
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[K_SPACE]:
                    print("recognized gesture: " + predicted_gest)
                    # call to the polly
                    if not is_voiceless:
                        text_to_polly_sound(predicted_gest)
                        is_voiceless = True
        '''
        else:
            print("PAUSE! GEST IS FINE")
            print("last predicted gest: " + predicted_gest)
            # call to the polly
            if not is_voiceless:
                text_to_polly_sound(predicted_gest)
                is_voiceless = True
        '''

    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)

    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand

    cv2.putText(frame_copy, "Hand sign recognition...", (10, 20), cv2.FONT_ITALIC, 0.7, (50, 50, 50), 1)
    cv2.imshow("Sign Detection", frame_copy)

    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()
