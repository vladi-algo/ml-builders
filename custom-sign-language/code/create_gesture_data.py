import cv2
import time
import numpy as np
import sys, getopt

background = None
accumulated_weight = 0.5

# current gesture to be generated
# element = 0
# num_frames = 0
# num_imgs_taken = 0


ROI_top = 100
ROI_bottom = 1000  # height
ROI_right = 100
ROI_left = 750  # width


def cal_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours for the image
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:

        hand_segment_max_cont = max(contours, key=cv2.contourArea)

        return (thresholded, hand_segment_max_cont)


def init(argv):
    if len(argv) == 0:
        raise getopt.GetoptError(
            "Invalid input, must be: create_gesture_data.py -i <image id= 0,1,2,...> -g <train/test>")
    try:
        opts, args = getopt.getopt(argv, "i:g:")
    except getopt.GetoptError:
        print("create_gesture_data.py -i <image id= 0,1,2,...> -g <train/test>")
        sys.exit(2)

    for opt, arg in opts:
        print(f" opt={opt} ,arg={arg} ")
        if opt == '-i':
            element = arg
        elif opt == '-g':
            group = arg
        else:
            sys.exit(-1)
    return (element, group)

def move_window(text):
    named_window = text
    cv2.namedWindow(named_window)
    cv2.moveWindow(named_window, 850, 120)
    return named_window


def generate_gestures(argv):
    element, group = init(argv)

    print(f'Generating images for image {element}, group {group} ...')
    cam = cv2.VideoCapture(1)
    num_frames = 0
    num_imgs_taken = 0

    while True:
        ret, frame = cam.read()

        # filpping the frame to prevent inverted image of captured frame...
        frame = cv2.flip(frame, 1)

        frame_copy = frame.copy()

        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        # convert to B/W
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        if num_frames < 200:
            cal_accum_avg(gray_frame, accumulated_weight)
            if num_frames <= 199:
                cv2.putText(frame_copy, "FETCHING BACKGROUND. PLEASE WAIT...", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2)
                # cv2.imshow("Sign Detection",frame_copy)

        # Time to configure the hand specifically into the ROI...
        elif num_frames <= 400:

            hand = segment_hand(gray_frame)

            #cv2.putText(frame_copy, "Adjust hand...Gesture for " + str(element), (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #            (0, 0, 255), 2)

            # Checking if hand is actually detected by counting number of contours detected...
            if hand is not None:
                thresholded, hand_segment = hand

                named_window = move_window("Adjusting hand gesture...")

                # Draw contours around hand segment
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

                cv2.putText(frame_copy, str(num_frames) + "For gesture" + str(element), (60, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

                # Also display the thresholded image
                cv2.imshow(named_window, thresholded)

        else:

            # Segmenting the hand region...
            hand = segment_hand(gray_frame)

            # Checking if we are able to detect the hand...
            if hand is not None:

                # unpack the thresholded img and the max_contour...
                thresholded, hand_segment = hand

                # Drawing contours around hand segment
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

                #print total number of frames
                #cv2.putText(frame_copy, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(frame_copy, str(num_frames)+"For" + str(element), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                #print total number of taken images
                cv2.putText(frame_copy, str(num_imgs_taken) + ' images' + " for gesture: " + str(element), (60, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                named_window = move_window("Now is taking hand image ...")

                cv2.imshow(named_window, thresholded)
                if num_imgs_taken <= 300:
                    cv2.imwrite(r"../images/" + group + "/" + str(element) + "/gesture" + str(time.time()) + '.jpg',
                                thresholded)
                else:
                    break
                num_imgs_taken += 1
            else:
                cv2.putText(frame_copy, 'No hand detected...', (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Drawing ROI on frame copy
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)

        cv2.putText(frame_copy, "Hand sign recognition...", (10, 20), cv2.FONT_ITALIC, 0.5, (51, 81, 51),
                    1)

        # increment the number of frames for tracking
        num_frames += 1

        # Display the frame with segmented hand
        cv2.imshow("Sign Detection", frame_copy)

        # Closing windows with Esc key...(any other key with ord can be used too.)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    # Releasing camera & destroying all the windows...

    cv2.destroyAllWindows()
    cam.release()


if __name__ == "__main__":
    generate_gestures(sys.argv[1:])
