
import cv2

import numpy as np
import cv2
from matplotlib import pyplot as plt

frameQueueLength = 2;
frameQueue = []
img4 = 0
maxMatches = 10
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

    # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        img3 = cv2.drawMatches(frameQueue[0], kp1, frameQueue[1], kp2, matches[:maxMatches],img4,flags=2)
        coords = []

        # getting coordinates of matched points
        coords1 = [kp1[mat.queryIdx].pt for mat in matches]
        coords2 = [kp2[mat.trainIdx].pt for mat in matches]
            #print("x: %s y: %s "% (x,y))

        coords1 = np.float32(coords1)
        coords2 = np.float32(coords2)
        #RASNAC
        focal = 1.0
        E, mask = cv2.findEssentialMat(coords2, coords1, focal = 1.0, pp = (0.,0.), method = cv2.RANSAC, prob = 0.999, threshold = 1.0)
        #R, t = cv2.recoverPose(E, pcoords2, coords1, focal = 1.0, pp = (0.,0.), mask)
        points, R, t,mask = cv2.recoverPose(E, coords2, coords1)


        # drawing matched points
        for marker in coords1[:maxMatches]:
            img4 = cv2.drawMarker(img3, tuple(int(i) for i in marker), color=(0, 255, 255))

        cv2.imshow('frame', img4)

        # obtaining distance
        #dist_x = (coords1[0][0] - coords2[0][0])
        #dist_y = (coords1[0][1] - coords2[0][1])
        # if dist_x != 0:
        #print("distance: %d,%d" % (dist_x,dist_y) )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

