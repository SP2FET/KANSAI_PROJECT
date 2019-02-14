import cv2

cap = cv2.VideoCapture(0)


while(True):

    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(frame, (7,7))
    cv2.drawChessboardCorners(frame, (7,7), corners, ret)

    cv2.imshow('frame', frame)



    if cv2.waitKey(30) & 0xFF == ord('q'):
        break