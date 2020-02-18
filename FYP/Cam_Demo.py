# import the necessary packages
import os 
import cv2 
import time
import torch
import imutils
import argparse
from GoTrack import GOTURN
from imutils.video import FPS
from imutils.video import VideoStream

frameDim = 500

parser = argparse.ArgumentParser(description= "CAM TRACKING TEST")
parser.add_argument("-v", "--video",
                type=str, help="path to the input video")
parser.add_argument("-mw", "--model_weights", 
                    default='../weights/FYPpytorch_goturn.pth.tar', 
                type=str, help="path to the model weights")
parser.add_argument("-s", "--save-directory",
                    default="../FYP/Results")


def main(args):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
   
    
    if args.video == None:
        print("[INFO] starting video stream..")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
        
    else:
        print("[INFO] loading input video..")
        vs = cv2.VideoCapture(args["video"])
    
    fps = None
    initBB = None 
    print("[INFO] Loading model weights..")
    tracker = GOTURN(device, args.model_weights)
    print("[INFO] Model weights successfully loaded..")
    while True:        
        # get current frame from the input video
        frame = vs.read()
        frame = frame[1] if args.video != None else frame 
        
        if frame is None:
            break 
        
        # vary the size of the frame (if required process faster) and 
        # grab frame dimensions
        frame = imutils.resize(frame, width = frameDim)
        (H, W) = frame.shape[:2]
        
        #tracker = GOTURN(frame, initBB, args.model_weights, device)
        # check to see if we are currently tracking an object 
        
        if initBB is not None:
            
            box = tracker.track(frame)
            (x1, y1, x2, y2) = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            #update the fps counter
            fps.update()
            fps.stop()
            info = [("FPS", "{:.2f}".format(fps.fps()))]
            
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        # showing the tracking output on the frame 
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        
        # if the 's' key is selected, we are going to "select" a bounding 
        # box to track 
        if key == ord("s"):
            initSelect = cv2.selectROI("Frame", frame, fromCenter=False, 
                                   showCrosshair=True)
            x1 = initSelect[0]
            x2 = initSelect[0] + initSelect[2]
            y1 = initSelect[1]
            y2 = initSelect[1] + initSelect[3]
            initBB = [x1, y1, x2, y2]
            #print("CV2 format Coordinates: {}".format(initSelect))
            #print("GOT format Coordinates: {}".format(initBB))
            tracker.track(frame, initBB)
            fps = FPS().start()
        
        elif key == ord("q"):
            break
        
    if args.video == None:
        vs.stop()
        
    else: 
        vs.release()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #args = vars(parser.parse_args())
    args = parser.parse_args()
    main(args)
