import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
#######################
brushThickness = 25
eraserThickness = 100
########################

folderPath = "Header" #name of folder
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList: #accessing each image 
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)

print(len(overlayList)) #importing all images
header = overlayList[0] #initial
drawColor = (255, 0, 255) #default -> purple

cap = cv2.VideoCapture(0)
cap.set(3, 1280) #exact same size as overlay image
cap.set(4, 720)

detector = htm.handDetector(maxHands=1,detectionCon=0.65) #using HandTrackingModule
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8) #canvas on which we will draw

while True:
        # STEP 1. Import image
        success, img = cap.read()
        img = cv2.flip(img, 1) #flip image horizontally


        # STEP 2. Find Hand Landmarks
        img = detector.findHands(img) #detect hand
        lmList = detector.findPosition(img, draw=False) #landmark list

        if len(lmList) != 0:
            # tip of index and middle fingers (x&y co-ord)
            x1, y1 = lmList[8][1:]  # index (id=8)
            x2, y2 = lmList[12][1:]  # middle (id=12)

            
            # STEP 3. Check which fingers are up
            fingers = detector.fingersUp()  # calling fingers module
            #print(fingers)

            
            # 4. If Selection Mode – Two fingers are up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0 #whenever we have a new selection (start from right pos instead of random pos)
                print("Selection Mode")
                ## Checking for the click if we are in the header
                if y1 < 125:  # value of header
                    if 250 < x1 < 450:
                        header = overlayList[0]  # brush purple selected -> img 0
                        drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:
                        header = overlayList[1]  # brush red selected -> img 1
                        drawColor = (255, 0, 0)
                    elif 800 < x1 < 950:
                        header = overlayList[2]  # brush green selected -> img 2
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]  # brush green selected -> img3
                        drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)  # for selection : rectangle

            # STEP 5. If Drawing Mode – Index finger is up
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)  # for drawing : circle
                print("Drawing Mode")

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0): #black color 
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x1, y1

        #merging image and image canvas(overlay) - so that we draw on the image
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY) #converting img canvas into gray image
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) #converting to binary image and inversing
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR) #converting it back to original
        img = cv2.bitwise_and(img, imgInv) #bitwise AND of original and inverse image
        img = cv2.bitwise_or(img, imgCanvas) #bitwise OR - final image

        # Setting the header image
        img[0:125, 0:1280] = header  # slicing image range of ht, range of width
        # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
        cv2.imshow("Image", img)
        cv2.imshow("Canvas", imgCanvas)
        cv2.waitKey(1)


