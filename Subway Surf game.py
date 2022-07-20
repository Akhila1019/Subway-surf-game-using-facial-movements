# Importing necessary libraries
import cv2
import keyboard
import math
import numpy as np
import pyautogui

    # this function is used to detect faces from webcam
def detectFaces(frame, faceHaarcascade):
    centreFaceCoordinates = []
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # to detect faces, image must be in gray scaled
                                                               # grayscale simplifies the algorithm and reduces computational requirements
    trained_data = cv2.CascadeClassifier(faceHaarcascade)   # load pretrained models
    face_coordinates = trained_data.detectMultiScale(grayscaled_img)   # returns the coordinates of the face
    for (x, y, w, h) in face_coordinates:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)   will draw rectangle around the face
        cv2.circle(frame, ((2*x+w)//2, (2*y+h)//2), 3,(0,255,0),-1)   # will draw a circle on the centre of the face
                                                                      # x+(w//2), y+(h//2)
    centreFaceCoordinates = [(2*x+w)//2, (2*y+h)//2]
    return frame, centreFaceCoordinates

    # this function will draw the rectangles
def drawRectangles(frame, frame_w, frame_h):
    centre_w = frame_w//2 - 120
    centre_h = frame_h//2
    size = 55
    x1 = centre_w - size
    y1 = centre_h - size
    x2 = centre_w + size
    y2 = centre_h + size

    cv2.rectangle(frame, (int(centre_w-size), int(centre_h-size)), (int(centre_w+size), int(centre_h+size)), (0, 0, 0), 2)   # centre rectangle
    cv2.rectangle(frame, (int(centre_w-3*size), int(centre_h-size)), (int(centre_w-size), int(centre_h+size)), (0, 0, 0), 2)   # left rectangle
    cv2.rectangle(frame, (int(centre_w-size), int(centre_h-3*size)), (int(centre_w+size), int(centre_h-size)), (0, 0, 0), 2)   # top rectangle
    cv2.rectangle(frame, (int(centre_w+size), int(centre_h-size)), (int(centre_w+3*size), int(centre_h+size)), (0, 0, 0), 2)   # right rectangle
    cv2.rectangle(frame, (int(centre_w-size), int(centre_h+size)), (int(centre_w+size), int(centre_h+3*size)), (0, 0, 0), 2)   # bottom rectangle

    return [(x1, y1), (x2, y2)]   # returns the coordinates of centre rectangle

    # this function is used to detect the direction
def detectDirection(centreFaceCoords, centreRectangleCoords, cmd):
    [(x1, y1), (x2, y2)] = centreRectangleCoords
    xc, yc = centreFaceCoords

    if xc < x1:
        cmd = 'left'
    elif xc > x2:
        cmd = 'right'
    elif yc < y1:
        cmd = 'up'
    elif yc > y2:
        cmd = 'down'

    if cmd:
        print('command: ', cmd, '\n')
        keyboard.press_and_release(cmd)   # this will press and release the given key
        return cmd

    # this function is used to change the flag
def changeFlag(centreFaceCoords, centreRectangleCoords, cmd):
    [(x1, y1), (x2, y2)] = centreRectangleCoords
    xc, yc = centreFaceCoords

    if x1 < xc < x2 and y1 < yc < y2:
        return True, None   # returns true when the face is detected in the centre rectangle
    return False, cmd   # returns false when the face is detected in other rectangles

def detectGesture(frame):

    cv2.rectangle(frame, (620, 380), (380, 90), (0, 255, 0), 0)
    croppedFrame = frame[100:380, 390:600]
    grayCroppedFrame = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)

    value = (35, 35)
    blurredCroppedFrame = cv2.GaussianBlur(grayCroppedFrame, value, 0)
    _, thresholdFrame = cv2.threshold(blurredCroppedFrame, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, heirarchy = cv2.findContours(thresholdFrame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(cnt)

    cv2.rectangle(croppedFrame, (x, y), (x+w, y+h), (0, 0, 255), 0)

    hull = cv2.convexHull(cnt)

    drawing = np.zeros(grayCroppedFrame.shape, np.uint8)

    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0

    cv2.drawContours(thresholdFrame, contours, -1, (0, 255, 0), 3)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        a = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        b = math.sqrt((far[0]-start[0])**2 + (far[1]-start[1])**2)
        c = math.sqrt((end[0]-far[0])**2 + (end[1]-far[1])**2)

        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        if angle <= 90:
            count_defects += 1
            cv2.circle(croppedFrame, far, 1, [0, 0, 255], -1)

        cv2.line(croppedFrame, start, end, [0, 255, 0], 2)

    cmd_hand = ''

    if count_defects == 0: # 1 for start and 2 for skateboard
        cmd_hand = 'space'
        pyautogui.click(clicks=2)
    elif count_defects == 4:
        cmd_hand = 'esc'

    if cmd_hand:
        print('command: ', cmd_hand, '\n')
        keyboard.press_and_release(cmd_hand)  # this will press and release the given key
        return cmd_hand


webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # captures video from webcam
width = webcam.get(3)   # Width of the video frame. can also use (cv2.CAP_PROP_FRAME_WIDTH)
height = webcam.get(4)  # Height of the video frame. can also use (cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)
faceHaarcascade = 'haarcascade_frontalface_default.xml'   # Detection Algorithm to detect faces
cv2.namedWindow('Face Detector', cv2.WINDOW_NORMAL)   # creates a window that can be used as a placeholder for images
                                                      # cv2.WINDOW_NORMAL : Allows to manually change window size
command = ''   # the direction to turn
flag = False

while True:
    is_successful, frame = webcam.read()   # reads each frame of the video and returns the frame and whether frame was successfully read or not
    frame = cv2.flip(frame, 1)   # flipping the image horizontally
    detectedFrame, centreFaceCoordinates = detectFaces(frame, faceHaarcascade)   # detect faces and returns the coordinates of the face
    cv2.putText(detectedFrame, command, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)   # prints the direction to turn from the facial movement
    centreRectangleCoordinates = drawRectangles(detectedFrame, width, height)   # returns the coordinates of the centre rectangle

    # cv2.rectangle(detectedFrame, (620, 380), (380, 90), (0, 0, 0), 2)
    # croppedFrame = detectedFrame[90:380, 380:620]
    command_hand = detectGesture(frame)

    if flag:
        command = detectDirection(centreFaceCoordinates, centreRectangleCoordinates, command)   # returns the direction
    flag, command = changeFlag(centreFaceCoordinates, centreRectangleCoordinates, command)
    # print(flag)
    cv2.imshow('Face Detector', frame)

    key = cv2.waitKey(1)
    if key == 81 or key == 113:   # press 'q' to exit
        break

webcam.release()
cv2.destroyAllWindows()