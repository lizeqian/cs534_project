#!/usr/bin/python

import numpy as np
import os
import cv2
import random

def random_select_frames(dir_v,video,dir_i):#input a video
    cap = cv2.VideoCapture(dir_v+video)
    frame_sum = cap.get(7)
    selected_frames = np.arange(int(0.3*frame_sum), int(frame_sum*0.9), 300) #random.sample(range(int(frame_sum*0.1), int(frame_sum*0.9)),10)
    
    for element in selected_frames:
        cap.set(1, element);
        ret, frame = cap.read()
        frame = cv2.resize(frame, (480, 270))
        if ret:
            img_name=dir_i+video+"_"+str(element)+".png"
            cv2.imwrite(img_name,frame)
        else:
            print "video cannot open this frame"
            continue
        
    
    cap.release()


if __name__ == "__main__":
    files= os.listdir("./video")
    
    for element in files:
        print "start video %s"%(element)
        directory = "./video_frames/"+str(element)+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        random_select_frames("./video/",element,directory)
        
        print "complete video %s"%(element)
    