
import cv2

import numpy as np
import cv2
from matplotlib import pyplot as plt

frameQueueLength = 2;
frameQueue = []
img4 = 0

cap = cv2.VideoCapture(0)

while(True):

    #CAPTTYRE IMAGE FRAMES
    ret, frame = cap.read()

    # ADDING KEY POINTS TO KEY POINTS QUEUE
    frameQueue.append(frame)
    if len(frameQueue) > frameQueueLength:
        del frameQueue[0]


    #FINDING KEYPOINTS
    orb = cv2.ORB_create()
    # kp = orb.detect(frame, None)
    if len(frameQueue) >= frameQueueLength:
        kp1, des1 = orb.detectAndCompute(frameQueue[0], None)
        kp2, des2 = orb.detectAndCompute(frameQueue[1], None)



    #EXTRACTION AND PRINT OF X,Y FROM KEYPOINTS
    # for idx in range(len(kp)):
    #     x = kp[idx].pt[0]
    #     y = kp[idx].pt[1]
    #
    #     print("x: %s y: %s "% (x,y))

    #DRAWING POINTS
    # img2 = frame.copy()
    # for marker in kp:
    #     img2 = cv2.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(0, 255, 255))

    # Display the resulting frame
    #cv2.imshow('frame',img2)
    # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: 1/(x.distance + 1))

        # Draw first 10 matches.
        img3 = cv2.drawMatches(frameQueue[0], kp1, frameQueue[1], kp2, matches[:10],img4,flags=2)

        cv2.imshow('frame', img3)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

