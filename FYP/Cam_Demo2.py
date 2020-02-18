# import the necessary packages
import os 
import cv2 
import time
import torch
import imutils
import argparse
import numpy as np
from GoTrack import GOTURN
from imutils.video import FPS
from imutils.video import VideoStream
 
""" mousedown = False 
mouseup = False
box = np.zeros(4) 
frameDim = 500 """

drawnBox = np.zeros(4)
boxToDraw = np.zeros(4)
mousedown = False
mouseupdown = False
initialize = False

def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, boxToDraw, initialize
    if event == cv2.EVENT_LBUTTONDOWN:
        drawnBox[[0,2]] = x
        drawnBox[[1,3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize = True
    boxToDraw = drawnBox.copy()
    boxToDraw[[0,2]] = np.sort(boxToDraw[[0,2]])
    boxToDraw[[1,3]] = np.sort(boxToDraw[[1,3]])

parser = argparse.ArgumentParser(description= "CAM TRACKING TEST")
parser.add_argument("-v", "--video",
                type=str, help="path to the input video")
parser.add_argument("-mw", "--model_weights", 
                    default='../weights/FYPpytorch_goturn.pth.tar', 
                type=str, help="path to the model weights")
parser.add_argument("-s", "--save-directory",
                    default="../FYP/Results")



def main(args, mirror=False):
    global tracker, initialize
    
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print("[INFO] Loading model weights..")
    tracker = GOTURN(device, args.model_weights)
    print("[INFO] Model weights successfully loaded..")
    
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam', 500, 500)
    cv2.setMouseCallback('Webcam', on_mouse, 0)
    frameNum = 0
    outputDir = None
    outputBoxToDraw = None
    
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        origImg = img.copy()
        if mousedown:
            cv2.rectangle(img,
                    (int(boxToDraw[0]), int(boxToDraw[1])),
                    (int(boxToDraw[2]), int(boxToDraw[3])),
                    [0,0,255], 2)
        elif mouseupdown:
            if initialize:
                outputBoxToDraw = tracker.track(img[:,:,::-1], boxToDraw)
                initialize = False
            else:
                outputBoxToDraw = tracker.track(img[:,:,::-1])
            cv2.rectangle(img,
                    (int(outputBoxToDraw[0]), int(outputBoxToDraw[1])),
                    (int(outputBoxToDraw[2]), int(outputBoxToDraw[3])),
                    [0,0,255], 2)
        cv2.imshow('Webcam', img)
        
        keyPressed = cv2.waitKey(1)
        if keyPressed == 27 or keyPressed == 1048603:
            break  # esc to quit
        frameNum += 1
    cv2.destroyAllWindows()
   
   
""" def DrawBox(event, x, y, flags, params):
    global mousedown, mouseup, box
    if event == cv2.EVENT_LBUTTONDOWN:
        box[[0,2]] = x 
        box[[1,3]] = y 
        mousedown = True
        mouseup = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        box[2] = x
        box[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        box[2] = x
        box[3] = y
    
    finalBox = box.copy()
    finalBox[[0,2]] = np.sort(finalBox[[0,2]])
    finalBox[[1,3]] = np.sort(finalBox[[1,3]])
    
    return finalBox     """


if __name__ == "__main__":
    #args = vars(parser.parse_args())
    args = parser.parse_args()
    main(args, mirror=True)
    #show_webcam(mirror=True)
