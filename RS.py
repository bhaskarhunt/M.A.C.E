#import liberaries

import numpy as np
import cv2
import sys
from skimage import data, io
from matplotlib import pyplot as plt

# Take input from terminal

if len(sys.argv) > 1:
    video = sys.argv[1]
    cap = cv2.VideoCapture(sys.argv[1])
    print("Given a desired input")
else :
    cap = cv2.VideoCapture('devel01/K_1.avi') 

#Variables used
_, frame = cap.read()
height, width, depth = frame.shape
print(height, width)
rx = int(height/2)
ry = int(width/2)
frame_list = []
difference_list = []
intersection_list = []
show = []
mask_list = []
fgbg = cv2.createBackgroundSubtractorMOG2() 


#Loop Starts

while(cap.isOpened()):
    _, frame = cap.read()
    #sframe = cv2.resize(frame, (rx, ry))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #depth_frame = frame[:, : ,1]
    GBgray_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)
    frame_list.append(GBgray_frame)
    if len(frame_list)>10:
        i = 0
        union_difference = np.zeros([height,width], dtype = int)

        #print(" union difference shape",union_difference.shape)
        while(i<len(frame_list)-1):
            difference = cv2.subtract(frame_list[i+1] ,frame_list[i])
            #m = difference.mean()
            _, difference = cv2.threshold(difference, 10, 255, cv2.THRESH_BINARY )
            kernel = np.ones((15,15), np.uint8)
            #difference = cv2.dilate(difference, kernel, iterations=1) 
            difference = cv2.erode(difference, kernel, iterations=1)
            kernel = np.ones((15, 15), np.uint8) 
            difference = cv2.dilate(difference, kernel, iterations=1) 
            difference_list.append(difference)
            union_difference = np.bitwise_or((union_difference),(difference))
            union_difference = np.uint8(union_difference)
            i += 1 
        frame_list.pop(0)


        binary = union_difference
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary)#connectivity=4)
        sizes = stats[:, -1]
        # # mask = np.array(nb_components, dtype=np.uint8)
        mask = np.array(output, dtype=np.uint8)
        # print(output.shape)
        max_label = 1
        try:
            max_size = sizes[1]
        except:
            print("no movement detected")
        for i in range(2, nb_components):
            try:
                if sizes[i] > max_size:
                    max_label = i
                    max_size = sizes[i]
            except:
                print("no movement detected")
        #print("maximum level---------",max_label)
        img2 = np.zeros(output.shape)
        img2[output == max_label] = 255
        
        # print(centroids[max_label-1])
        # a, b= centroids[max_label-1]
        # a = np.uint8(a)
        # b = np.uint8(b)
        cv2.imshow("Biggest component", img2)
        # frame[a:a+5, b:b+5] = [0, 0, 255] 
        # cv2.imshow("frame3", frame)
        # plt.hist(img2.ravel(),256,[0,256]); plt.show()


        # _, labels = cv2.connectedComponents(union_difference)
        # print(labels.shape)
        # mask = np.array(labels, dtype=np.uint8)
        # mask[labels == 1] = 255
        # print(mask)
        # mask_list.append(mask)
        # _, img2 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY )
        i = 0
        # cv2.imshow("frame2", difference_list[3])
        #print(max(difference_list[0]), max(img2))
        while(i<len(frame_list)):
            intersection_list.append(np.bitwise_and(img2.astype(int),difference_list[i].astype(int)))
            # print(str(type(img2))+" and "+str(type(difference_list[i])))
            i += 1 
            print(i)
        #print(len(intersection_list))
        try:
            contours,hierarchy = cv2.findContours(np.uint8(intersection_list[9]), 1, 2)
            cnt = contours[0]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            frame = cv2.drawContours(frame,[box],0,(0,0,255),2)
        except :
            print("no conture found")
        cv2.imshow("intersection2", np.uint8(intersection_list[1]))
        # from scipy import ndimage
        # [x, y] = ndimage.measurements.center_of_mass(intersection_list[4])
        # x = np.uint8(x)
        # y = np.uint8(y)
        # frame[x:x+5, y:y+5] = [0, 0, 255]
        intersection_list.clear()
        difference_list.clear()
    
        cv2.imshow("frame", mask)
        cv2.imshow("frame2",frame)
    k = cv2.waitKey(300)
    if k == 27:
        break
 
cv2.destroyAllWindows()
cap.release()