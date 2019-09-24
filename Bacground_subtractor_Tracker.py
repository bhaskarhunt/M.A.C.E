import numpy as np
import cv2
import sys
import time
clk_id = time.time() 
if len(sys.argv) > 1:
    video = sys.argv[1]
    cap = cv2.VideoCapture(sys.argv[1])

else :
    cap = cv2.VideoCapture('K_1.avi') 
i = 0
X = []
while i<4:
    _, frame = cap.read()
    frame = frame[:, :, 1]
    #_, frame = cv2.threshold(frame, 140, 255, cv2.THRESH_BINARY_INV )
    X.append(frame)
    i += 1
X = np.array(X)
centroid = []
avg1 = np.float32(frame)
path = []
Z = []
t1 = 0
while(cap.isOpened()):
    _, frame = cap.read()
    distanceframe = frame[:, :, 1]
    distanceframe = cv2.GaussianBlur(distanceframe, (55, 55), 0)
    print(X.shape)
    m = distanceframe.mean()
    cv2.accumulateWeighted(distanceframe,avg1,0.5)
    res1 = cv2.convertScaleAbs(avg1)
    res1 = cv2.GaussianBlur(res1, (55, 55), 0)
    diff = cv2.absdiff(distanceframe, res1)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY ) 
    print(X.shape)
    from scipy import ndimage
    [x, y] = ndimage.measurements.center_of_mass(diff)
    if not np.isnan(x) == True:
        t = time.time()
        dt = t - t1
        x = int(np.round_(x))
        y = int(np.round_(y))
        Z = [x, y, dt]
        t1 = t
        path.append(Z)
        print(path[0])
        if len(path) > 5:
            path.pop(0)
        for i in range(len(path)):
            [a, b, tim] = path[i]
            frame[a:a+5, b:b+5] = [0, 0, 255]
        print(path)

    cv2.imshow("frame", diff)
    cv2.imshow("video", frame)
    X = np.array([X[1,:,:],X[2,:,:],X[3,:,:],distanceframe])
    #print(X.shape)
    k = cv2.waitKey(100)
    if k == 27:
        break
 
cv2.destroyAllWindows()
cap.release()