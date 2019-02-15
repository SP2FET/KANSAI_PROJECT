import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt

img4 = 0
maxMatches = 50
cap = cv2.VideoCapture("krstn.mov")

ret, frame1 = cap.read()
orb = cv2.ORB_create(10)

kp1 = orb.detect(frame1, None)
kp1, des1 = orb.compute(frame1, kp1)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def cls():
    os.system("clear")

#position of a camera
#dynamic plot
fig = plt.figure()
ax = fig.add_subplot(111)
Ln, = ax.plot([0,0])
ax.set_xlim([-1000,1000])
ax.set_ylim([-1000,1000])
plt.ion()
plt.show()

x = []
y = []

R_pos = [[1,0,0],[0,1,0],[0,0,1]]
t_pos = [[0],[0],[0]]
scale = 1
focal = 718.8560
pp = (607.1928, 185.2157)

while(True):

    #CAPTTYRE IMAGE FRAMES
    ret, frame2 = cap.read()
    kp2, des2 = orb.detectAndCompute(frame2, None)


    #FINDING KEYPOINTS
    orb = cv2.ORB_create()

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:maxMatches],None,flags=2)
    coords = []

    # getting coordinates of matched points
    coords1 = [kp1[mat.queryIdx].pt for mat in matches]
    coords2 = [kp2[mat.trainIdx].pt for mat in matches]

    coords1 = np.float32(coords1)
    coords2 = np.float32(coords2)

    if coords1.size > 10 and coords2.size > 10:

        # RASNAC
        E, mask = cv2.findEssentialMat(coords2, coords1, focal, pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # R, t = cv2.recoverPose(E, pcoords2, coords1, focal = 1.0, pp = (0.,0.), mask)
        points, R, t, mask = cv2.recoverPose(E, coords2, coords1)


        t_pos = t_pos + scale * np.dot(R_pos,t)
        R_pos = np.dot(R, R_pos)

        # drawing matched points
        for marker in coords1[:maxMatches]:
            img4 = cv2.drawMarker(img3, tuple(int(i) for i in marker), color=(0, 255, 255))


        y.append(t_pos[1])
        x.append(t_pos[0])

        if min(x) < min(y):
            mindim = min(x)
        else:
            mindim = min(y)

        if max(x) > max(y):
            maxdim = max(x)
        else:
            maxdim = max(y)

        ax.set_xlim([mindim - 5.0, maxdim + 5.0])
        ax.set_ylim([mindim - 5.0, maxdim + 5.0])

        Ln.set_ydata(y)
        Ln.set_xdata(x)
        plt.pause(0.003)

        cv2.waitKey(1)

        frame1 = frame2.copy()
        kp1, des1 = kp2.copy(), des2.copy()


    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', img3)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()