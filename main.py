import cv2
import os
import argparse
import time
from segpipeline import SegmentationPipeline
        
def process_batch(dir, input_dir, p):
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


def process_single(image_path, p):      
    image_name, _ = os.path.basename(image_path).split(".")
    output_dir = os.path.dirname(image_path)
    
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
    

if __name__ == "__main__":
    
    dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename")
    parser.add_argument("--input_dir")
    args = parser.parse_args()
    p = SegmentationPipeline()  
   
    if args.filename is None:
        input_dir = args.input_dir
        process_batch(dir, input_dir, p)
                
    else:
        image_path = args.filename
        process_single(image_path, p)

