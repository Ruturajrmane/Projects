import cv2
import pickle
import cvzone
import numpy as np


with open('Carpark', 'rb') as f:
    poslist = pickle.load(f)

def checkParkingsapce(imgdilate):

    counter = 0
    for pos in poslist:
        x,y = pos

        croppedimg = imgdilate[y:y+height, x:x+width]
        # cv2.imshow(str(x*y), croppedimg)
        # Black pixel represented by 0 and white will be represented by 1
        count = cv2.countNonZero(croppedimg)
        cvzone.putTextRect(img, str(count), (x,y+height-10), scale= 1, thickness= 2, offset=0)

        if count < 900:
            color = (0,255,0)
            thickness = 5
            cv2.rectangle(img, (pos[0], pos[1]), (pos[0] + width, pos[1] + height), color, thickness)
            counter +=1
        else:
            color = (0,0,255)
            thickness = 2
            cv2.rectangle(img, (pos[0], pos[1]), (pos[0] + width, pos[1] + height), color,thickness)

        cvzone.putTextRect(img, str(counter), (100,50), scale=4, thickness=3, offset=0, colorR = (0,255,0))

width, height = 107,48

# Reading the video
cap = cv2.VideoCapture('Data/carPark.mp4')

while True:

    # To run the video in loop logic  --- (current frame == last frame then frame is set to zero)
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success, img = cap.read()

    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Larger kernels have more values factored into the average, and this implies that a larger kernel
    # will blur the image more than a smaller kernel. Kernel should be always odd to keep the pixel being
    # processed at center
    kernelb = (5,5)
    # Blur applied to smooth out the transition from one side of an edge to another
    imgblur = cv2.GaussianBlur(imggray, (5,5), 1)
    # To change image from gray to binary image i.e black and white
    imgthresh = cv2.adaptiveThreshold(imgblur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 25,16)

    imgmed = cv2.medianBlur(imgthresh, 5)
    kernel = np.ones((3,3),np.int8)
    # Dilate increases the edge thickness
    imgdilate = cv2.dilate(imgmed, kernel, iterations = 1)

    checkParkingsapce(imgdilate)

    cv2.imshow('Image', img)
    # cv2.imshow('Imageblur', imgblur)
    # cv2.imshow('Imagethresh', imgthresh)
    # cv2.imshow('Imagemde', imgmed)
    cv2.waitKey(10)