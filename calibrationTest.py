import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# size of the chessboard
objp = np.zeros((7*7,3), np.float32)
# real points on the chess board( in mm's)
objp[:,:2] = np.mgrid[0:7*0.24:0.24,0:7*0.24:0.24].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# image selection part
while(True):

    ret, frame = cap.read()
    frameCpy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects the corners on the chessboard
    ret, corners = cv2.findChessboardCorners(gray, (7,7))
    if(ret == True):
        # if detected -> draw the frame with detected corners
        cv2.drawChessboardCorners(frame, (7,7), corners, ret)
        cv2.imshow('frame', frame)

        # when you press 's' the frame will be taken into account in calibration
        if cv2.waitKey(3) & 0xFF == ord('s'):
            pass
            # save current frame to teaching vector - or jut use it here t calibrate
            cv2.imshow('to calibration', gray)
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # reason of this is not to give same frames to calibration but to select only some of these

    # when You have enough frames for calibration just end collecting frames by pressing 'q'
    # You need around 15 frames to make calibration work
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
# calibration part
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# test of undistortion
while(True):
    ret, frame = cap.read()
    cv2.imshow('raw image', frame)

    h,  w = frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]

    cv2.imshow('undistorted image', dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# undistorting a prepared image
# frame = cv2.imread('chessboard.png')
# h,  w = frame.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#
# # undistort
# dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
#
# # crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('undistorted.png', dst)
# # cv2.imshow('undistorted',dst)

sth = input('stop: ')

