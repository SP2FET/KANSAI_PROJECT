import cv2
import numpy as np

def calibrate():
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

    # write calibrated matrixes and vectors to file
    np.savetxt('mtx.txt', mtx, fmt='%.2f')
    np.savetxt('dist.txt', dist, fmt='%.2f')

    with open('rvecs.txt', 'w') as f:
        for item in rvecs:
            f.write("%s\n" % item)
        f.close()

    with open('tvecs.txt', 'w') as f:
        for item in tvecs:
            f.write("%s\n" % item)
        f.close()



def readCalibration(cap):
    # run this script after calibration when You have files mtx.txt, dist.txt, rvecs.txt, tvecs.txt

    mtx = np.loadtxt('mtx.txt')
    dist = np.loadtxt('dist.txt')

    with open('rvecs.txt') as f:
        rvecs = f.read().splitlines()
        f.close()

    with open('tvecs.txt') as f:
        tvecs = f.read().splitlines()
        f.close()

    # adjust crop to the used camera
    ret, frame = cap.read()

    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # x, y, w, h = roi

    return (mtx, dist, rvecs, tvecs, newcameramtx, roi)



def undistortAndCrop(frame, camParams):
    # give raw frame to this frame and receive frame ready to dehazing/other stuff
    # camParams should be obtained from calibration

    mtx, dist, rvecs, tvecs, newcameramtx, roi = camParams
    x, y, w, h = roi

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image
    dst = dst[y:y+h, x:x+w]

    # turn it gray
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    return gray


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    # no need to run calibrate() if You have calibrated camera once
    calibrate()
    camParams = readCalibration(cap)

    while(True):

        ret, frame = cap.read()

        frame = undistortAndCrop(frame, camParams)


        cv2.imshow('frame', frame)

        # to stop the test press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

