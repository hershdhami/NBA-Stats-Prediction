import cv2
import time
import numpy as np
import mediapipe as mp

########################################
wCam, hCam = 640, 480
########################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

while True:
    success, img = cap.read()

    # Time when we finish processing this frame
    cTime = time.time()


    fps = 1 / (cTime - pTime)
    pTime = cTime

    #Expression to be evaluated is put in the brackets for fprefix string#
    cv2.putText(img, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 255), 3)
    cv2.imshow("Img",img)
    cv2.waitKey(1)

    #Press 'Q' if you want to exit the frame
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

# When everything done, release the capture
cap.release()

# Destroy the all windows now
cv2.destroyAllWindows()