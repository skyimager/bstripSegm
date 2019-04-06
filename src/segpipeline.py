import cv2
import numpy as np
from src.img_utils import plthist, crop_center

class SegmentationPipeline(object):
    def __init__(self):
        pass
    
    def segmented_strip(self,image):    
        # Otsu Thresholding with blur
        blur1 = cv2.GaussianBlur(image,(7,7),0)
        bw = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
        ret, otsu_th = cv2.threshold(bw,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
        # Morphological Operation
        # Area closing
        kernel = np.ones((120,120),'uint8')
        closing = cv2.morphologyEx(otsu_th, cv2.MORPH_CLOSE, kernel)
    
        # Preparing segmented image
        mask = cv2.merge((closing,closing,closing))
        final= cv2.bitwise_and(image, mask)
        
        return final
    
    def segmented_bacteria(self,image):
        height, width = image.shape[0:2]
        # Crop from Center
        cropped = self.crop_center(image,width//2,height//3)
        
        # Lets check out HSV Channel
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0]
        
        # Global thresholding on hue channel
        ret, hue = cv2.threshold(h,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)     
       
        # Area opening on cropped image
        kernel = np.ones((23,23),'uint8')
        opening = cv2.morphologyEx(hue, cv2.MORPH_OPEN, kernel)
        
        # Final dilation
        kernel = np.ones((3,3),'uint8')
        dilate = cv2.dilate(opening,kernel,iterations=2)
      
        # Convert the image back to original shape
        top = height//3
        left = width//4
        constant= cv2.copyMakeBorder(dilate,top,top,left,left,cv2.BORDER_CONSTANT,0)        
       
        mask = cv2.merge((constant,constant,constant))
        final2 = cv2.bitwise_and(image, mask)
        
        return final2
    
    def autocrop(self,strip):
        tol =0
        mask = strip[:,:,0]>tol
        ch1 = strip[:,:,0][np.ix_(mask.all(1),mask.any(0))]
        ch2 = strip[:,:,1][np.ix_(mask.all(1),mask.any(0))]
        ch3 = strip[:,:,2][np.ix_(mask.all(1),mask.any(0))]
        final = cv2.merge((ch1,ch2,ch3))
        return final