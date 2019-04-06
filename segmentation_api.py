import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time

class SegmentationPipeline(object):
    def __init__(self):
        pass
        
    def plthist(self,image):    
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([image],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        return plt.show()
    
    def crop_center(self, image,cropx,cropy):
        y,x = image.shape[0:2]
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return image[starty:starty+cropy,startx:startx+cropx,:]
    
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

if __name__ == "__main__":
    
    dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename")
    parser.add_argument("--input_dir")
    args = parser.parse_args()
    p = SegmentationPipeline()  
   
    if args.filename is None:
        input_dir = args.input_dir
        output = os.path.basename(input_dir)+"_output"
        output_dir = os.path.join(dir,output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)           
        for root, __, files in os.walk(input_dir):  
            for i, file in enumerate(files):
                start_time = time.time()
                image_path = os.path.join(root, file)
                print("Processing file at:", image_path)
                image_name, _ = os.path.basename(image_path).split(".")
                
                image = cv2.imread(image_path)     
                sv_name =  image_name + "_original.png"
                destination = os.path.join(output_dir,sv_name)
                cv2.imwrite(destination,image)
                
                strip = p.segmented_strip(image)
                sv_name =  image_name + "_segmented_strip.png"
                destination = os.path.join(output_dir,sv_name)
                cv2.imwrite(destination,strip)
                
                cropped = p.autocrop(strip)
                sv_name =  image_name + "_cropped_strip.png"
                destination = os.path.join(output_dir,sv_name)
                cv2.imwrite(destination,cropped)
                
                bacteria = p.segmented_bacteria(strip)    
                sv_name =  image_name + "_segmented_xbacteria.png"
                destination = os.path.join(output_dir,sv_name)
                cv2.imwrite(destination, bacteria) 
                print("Total time taken for " + os.path.basename(image_path)
                       +" " +str(time.time() - start_time))
                
    else:
        image_path = args.filename        
        image_name, _ = os.path.basename(image_path).split(".")
        
        image = cv2.imread(image_path)     
        sv_name =  image_name + "_original.png"
        cv2.imwrite(sv_name,image)
        
        strip = p.segmented_strip(image)
        sv_name =  image_name + "_segmented_strip.png"
        cv2.imwrite(sv_name,strip)

        cropped = p.autocrop(strip)
        sv_name =  image_name + "_cropped_strip.png"
        destination = os.path.join(output_dir,sv_name)
        cv2.imwrite(destination,cropped)
        
        bacteria = p.segmented_bacteria(strip)    
        sv_name =  image_name + "_segmented_xbacteria.png"
        cv2.imwrite(sv_name,bacteria)
