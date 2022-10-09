import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2,modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils #drawing lines
        self.tipIds = [4, 8, 12, 16, 20] #thumb, index, middle, ring, pinky

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #hands take in only rgb image
        self.results = self.hands.process(imgRGB)
        #print(self.results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks: #draw points - (21)
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS) #draw lines connecting each landmark
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]#info of particular hand number
            for id, lm in enumerate(myHand.landmark): #id of each landmark and landmark(x,y,z)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) #find pos of landmark on image
                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) #radius, rgb color, fill ; filling pos of landmark on hand with color

        return self.lmList

    def fingersUp(self):
        fingers = []
        # Thumb - FOR RIGHT HAND

        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1) #right - opened
        else:
            fingers.append(0) #left - closed


        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]: #tip of finger above other landmark which is 2 steps below it
                fingers.append(1) #finger opened - above the other landmark
            else:
                fingers.append(0) #finger closed - below

        totalFingers = fingers.count(1)
        return fingers


def main():
    pTime = 0 #past time
    cTime = 0 #current time
    cap = cv2.VideoCapture(0) #0 or 1
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) #frames per second
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3) #pos,font,scale,rgb color of frame notif,thickness

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
