#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt

def plthist(image):    
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([image],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    return plt.show()

def crop_center(image,cropx,cropy):
    y,x = image.shape[0:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return image[starty:starty+cropy,startx:startx+cropx,:]