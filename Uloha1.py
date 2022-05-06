import cv2
import pandas as pd
import colorsys
import numpy as np
from cv2 import approxPolyDP
from cv2 import arcLength

cap = cv2.VideoCapture(0)

# declaring global variables (are used later on)
clicked = False
first= False
auto=False
count=0
r = g = b = x_pos = y_pos = 0
bm = [0,0]
gm = [0,0]
rm = [0,0]
global click
click=1

# initialization of color parameters
text = 'a' 
text2 = 'a'

kernel = np.ones((5,5), np.uint8) 

lower_1 = np.array([0, 0, 0])
upper_1 = np.array([0, 0, 0])

lower_2 = np.array([0, 0, 0])
upper_2 = np.array([0, 0, 0])

br=0
br0=[]
br0test=[]
brtest=[]

global xs1, ys1, xs2, ys2

from numpy.linalg import norm

def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)

# function to get x,y coordinates of mouse double click
def draw_function(event, x, y,flags, param):
    if (event == cv2.EVENT_LBUTTONDBLCLK or auto==True):#EVENT_LBUTTONDBLCLK
        global bm, gm, rm, clicked, click
        b, g, r = frameG[y, x]
        # vector of chosen colors
        bm.append(int(b)) 
        gm.append(int(g)) 
        rm.append(int(r)) 
        #click+=1
        #if(click%2==0):
        clicked = True

# set mouse clisk callback
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_function)


while True:
    _, imageFrame = cap.read()
    # filtration of data
    frameG = cv2.GaussianBlur(imageFrame, (7,7), 0)#cv2.medianBlur(frame, 21)#
    frameG = cv2.erode(frameG, kernel, iterations = 2)
    frameG = cv2.dilate(frameG, kernel, iterations = 1)

    # data for brightness
    br0.append(br)
    br=brightness(imageFrame)
    dbr=np.abs(br-br0[-1])
    del br0[:-1]
    
    # looking for brightness change
    
    #first=False
    if(first==True and dbr>2):
        auto=True
        event=1
        flags=1
        param=1
        bm=[]
        gm=[]
        rm=[]
        x1=int(xs1)
        y1=int(ys1)
        draw_function(event, x1, y1, flags, param)
        x2=int(xs2)
        y2=int(ys2)
        draw_function(event, x2, y2, flags, param)
        auto=False
        
    if clicked:
        first=True

        # cv2.rectangle(image, start point, endpoint, color, thickness)-1 fills entire rectangle
        cv2.rectangle(imageFrame, (20, 20), (750, 60), (bm[np.size(bm)-1], gm[np.size(bm)-1], rm[np.size(bm)-1]), -1)

        # Creating text string to display( Color name and RGB values )
        text = ' R=' + str(rm[np.size(bm)-1]) + ' G=' + str(gm[np.size(bm)-1]) + ' B=' + str(bm[np.size(bm)-1])
        
        hsv = cv2.cvtColor(frameG, cv2.COLOR_BGR2HSV)
        h,s,v=colorsys.rgb_to_hsv(rm[np.size(bm)-1]/255, gm[np.size(bm)-1]/255, bm[np.size(bm)-1]/255)
        h=h*360

        text2 = ' H=' + str(round(h,2)) + ' S=' + str(round(s,2)) + ' V=' + str(round(v,2))

        lower_1 = np.array([(h-11)/360*179, (s-0.3)*255, (v-0.2)*255])
        upper_1 = np.array([(h+11)/360*179, (s+0.3)*255, (v+0.2)*255])
        mask = cv2.inRange(hsv, lower_1, upper_1) 

        # cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(imageFrame, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # For very light colours we will display text in black colour
        if r + g + b >= 600:
            cv2.putText(imageFrame, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(imageFrame, text2, (50, 80), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA) 
            
        #--------------------
        cv2.rectangle(imageFrame, (20, 20), (750, 60), (bm[np.size(bm)-2], gm[np.size(bm)-2], rm[np.size(bm)-2]), -1)

        hg,sg,vg=colorsys.rgb_to_hsv(rm[np.size(bm)-2]/255, gm[np.size(bm)-2]/255, bm[np.size(bm)-2]/255)
        hg=hg*360

        lower_2 = np.array([(hg-11)/360*179, (sg-0.3)*255, (vg-0.2)*255])
        upper_2 = np.array([(hg+11)/360*179, (sg+0.3)*255, (vg+0.2)*255])
        maskg = cv2.inRange(hsv, lower_2, upper_2) 
        #--------------------

        clicked = False
        
    hsv = cv2.cvtColor(frameG, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_1, upper_1) 
    mask2 = cv2.inRange(hsv, lower_2, upper_2) 
    
    #----------For 1. param----------
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    for cnt in contours:
        #approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if ((cv2.contourArea(cnt) > 1000)):
            x1, y1, width1, height1 = cv2.boundingRect(cnt)

            imageFrame=cv2.rectangle(imageFrame, (x1 , y1), (x1 + width1, y1 + height1),(0, 0, 255), 2)

            # drawing a point
            xs1=int(x1+width1/2)
            ys1=int(y1+height1/2)

            cv2.rectangle(imageFrame,(xs1,ys1),(xs1+2,ys1+2),(255,0,255),3)
            
            approx = cv2.approxPolyDP(cnt,0.03*cv2.arcLength(cnt,True),True)
            if len(approx) == 3:
                shape = "triangle"
            elif len(approx) == 4:
                shape = "square"
            elif len(approx) > 7:
                shape = "circle"
                
            cv2.putText(imageFrame, shape, (xs1+5,ys1+5), 2, 0.8, (255, 0, 0), 2, cv2.LINE_AA)    
            
    #---------For 2. param------------
    
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        #approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        
        if ((cv2.contourArea(cnt) > 1000)):
            
            x2, y2, width2, height2 = cv2.boundingRect(cnt)
            
            
            imageFrame=cv2.rectangle(imageFrame, (x2 , y2), (x2 + width2, y2 + height2),(0, 0, 255), 2)
            
            # drawing a point
            xs2=int(x2+width2/2)
            ys2=int(y2+height2/2)
            
            cv2.rectangle(imageFrame,(xs2,ys2),(xs2+2,ys2+2),(255,0,255),3)
            
            approx = cv2.approxPolyDP(cnt,0.03*cv2.arcLength(cnt,True),True)
            if len(approx) == 3:
                shape2 = "triangle"
            elif len(approx) == 4:
                shape2 = "square"
            elif len(approx) > 7:
                shape2 = "circle"
                
            cv2.putText(imageFrame, shape2, (xs2+5,ys2+5), 2, 0.8, (255, 0, 0), 2, cv2.LINE_AA)    
            
    cv2.rectangle(imageFrame, (20, 20), (325, 60), (bm[np.size(bm)-1], gm[np.size(bm)-1], rm[np.size(bm)-1]), -1)
    cv2.rectangle(imageFrame, (326, 20), (750, 60), (bm[np.size(bm)-2], gm[np.size(bm)-2], rm[np.size(bm)-2]), -1)
    cv2.putText(imageFrame, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)    
    cv2.putText(imageFrame, text2, (50, 80), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)    


    cv2.imshow("frame",imageFrame)
    cv2.imshow("mask",mask1)

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
