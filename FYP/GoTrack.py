import os
import time
import argparse
import re

import torch
import numpy as np
import cv2

from model import GoNet
from helper import NormalizeToTensor, Rescale, crop_sample, bgr2rgb
from boundingbox import BoundingBox

class GOTURN:
    """Tester for Webcam"""
    def __init__(self, device, model_path):
        super().__init__()        
        self.device = device
        self.transform = NormalizeToTensor()
        self.scale = Rescale((224, 224))
        self.model_path = model_path
        self.model = GoNet()
        self.frameCnt = 0
        self.prevImg = None 
        self.currImg = None 
        self.opts = None
        self.img = [] 
        self.tracking = False
        self.prevBB = None
       # self.gt = []
       # self.opts = None
       # self.curr_img = None
       # self.initBB = initBB
        checkpoint = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
        self.model.eval()
      
        self.x = []
       
       
    def track(self, image, initBB = None):
        
        if type(image) == str:
            image = cv2.imread(image)[:,:,::-1]
        else:            
            image = bgr2rgb(image.copy())
       
        if initBB is not None:
            self.prevBB = np.array(initBB)
            self.prevImg = image   
            #self.prev_rect = init_bbox
            self.tracking = True
            self.frameCnt += 1 
           
        elif self.tracking == True:
            self.currImg = image
            self.img = [self.prevImg, self.currImg]
            self.frameCnt += 1
            
            if self.frameCnt == 2:
                
                prevSample, prevOpts = crop_sample({'image': self.prevImg, 'bb': self.prevBB})
                currSample, currOpts = crop_sample({'image': self.currImg, 'bb': self.prevBB})
                prev_img = self.scale(prevSample, prevOpts)['image']
                curr_img = self.scale(currSample, currOpts)['image']
                sample = {'previmg': prev_img, 'currimg': curr_img}
                #self.curr_img = curr
                self.opts = currOpts
                sample = self.transform(sample)
                bb = self.get_rect(sample)
                self.prevBB = bb
                
            self.prevImg = self.currImg
            self.frameCnt -= 1
                
        return self.prevBB       
        
        

    def __getitem__(self, idx):
        """
        Returns transformed torch tensor which is passed to the network.
        """
        sample = self._get_sample(idx)
        return self.transform(sample)


    def get_rect(self, sample):
        """
        Regresses the bounding box coordinates in the original image dimensions
        for an input sample.
        """
        x1, x2 = sample['previmg'], sample['currimg']
        x1 = x1.unsqueeze(0).to(self.device)
        x2 = x2.unsqueeze(0).to(self.device)
        y = self.model(x1, x2)
        bb = y.data.cpu().numpy().transpose((1, 0))
        bb = bb[:, 0]
        bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])

        # inplace conversion
        bbox.unscale(self.opts['search_region'])
        bbox.uncenter(self.currImg, self.opts['search_location'],
                      self.opts['edge_spacing_x'], self.opts['edge_spacing_y'])
        return bbox.get_bb_list()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Show the Webcam demo.')
    args = parser.parse_args()
    print(args)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    tester = GOTURN(frame, initBB, args.model_weights, device)
    tester.test()
