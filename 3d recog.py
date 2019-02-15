import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt

img4 = 0
maxMatches = 50
cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
orb = cv2.ORB_create(10)

kp1 = orb.detect(frame1, None)
kp1, des1 = orb.compute(frame1, kp1)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def cls():
    os.system("clear")

#position of a camera
x,y =[0],[0]
#dynamic plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# Ln, = ax.plot([0,0])
# ax.set_xlim([-1000,1000])
# ax.set_ylim([-1000,1000])
# plt.ion()
plt.show()

R_pos = [[1,0,0],[0,1,0],[0,0,1]]
t_pos = [[0],[0],[0]]

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
        focal = 1.0
        E, mask = cv2.findEssentialMat(coords2, coords1, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # R, t = cv2.recoverPose(E, pcoords2, coords1, focal = 1.0, pp = (0.,0.), mask)
        points, R, t, mask = cv2.recoverPose(E, coords2, coords1)

        t_p = np.dot(t.T, R_pos)
        t_pos = t_pos + t_p.T
        R_pos = np.dot(R, R_pos)

        # print(R_pos)
        # print(t_pos)
        # print("===========================")

        # drawing matched points
        for marker in coords1[:maxMatches]:
            img4 = cv2.drawMarker(img3, tuple(int(i) for i in marker), color=(0, 255, 255))

        th_x = math.asin(R[2][1])
        th_y = math.atan(-R[2][0]/R[2][2])
        th_z = math.atan(-R[0][1]/R[1][1])


        print("\r{0} {1} {2}".format(th_x, th_y, th_z), end="")


        theta = [0.0, th_z]

        r = [0.0, 0.5]

        plt.polar(theta, r)


        # y.append(t_pos[1])
        # x.append(t_pos[0])
        #
        # # ax.set_xlim([-max(x), max(x)])
        # # ax.set_ylim([-max(y), max(y)])
        #
        # ax.set_xlim([-max(x) - 5, max(x) + 5])
        # ax.set_ylim([-max(y) - 5, max(y) + 5])
        #
        # Ln.set_ydata(y)
        # Ln.set_xdata(x)
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