<<<<<<< HEAD
<<<<<<< Updated upstream
import cv2 as opencv
=======
import cv2
>>>>>>> Stashed changes
=======
import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    orb = cv2.ORB_create()
    kp = orb.detect(frame, None)
    kp, des = orb.compute(frame, kp)


    # EXTRACTION AND PRINT OF X,Y FROM KEYPOINTS
    for idx in range(len(kp)):
        x = kp[idx].pt[0]
        y = kp[idx].pt[1]

        print (x)
        print (y)

    img2 = frame.copy()
    for marker in kp:
        img2 = cv2.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(255, 255, 255))

    # Display the resulting frame
    cv2.imshow('frame',img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
>>>>>>> master
