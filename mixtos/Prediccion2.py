
import numpy as np
import cv2
from glob import glob
from time import time
import joblib

img = cv2.imread("mixtos1.tif")
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hierarchy = hierarchy[0]

for component in zip(contours, hierarchy):
    currentContour = component[0]
    currentHierarchy = component[1]    
    if(cv2.contourArea(currentContour)>500 and cv2.contourArea(currentContour)<500000):
        if(currentHierarchy[2] < 50):
            x,y,w,h = cv2.boundingRect(currentContour)
            cv2.drawContours(img,[currentContour],0,(0,0,255),3)
            M = cv2.moments(currentContour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            A = cv2.contourArea(currentContour)
            p = cv2.arcLength(currentContour,True)
            Comp = A/float(p*p) 
            RA = w/float(h) 
            Hu = cv2.HuMoments(M)
            
            VectorCarac = np.array([[A,p,Comp,RA,Hu[0],Hu[1],Hu[2],Hu[3],Hu[4],Hu[5],Hu[6]]], dtype = np.float32)
            
            modeloSVM = joblib.load('ModeloSVM_S2.sav')
            modeloS = joblib.load('ModeloStandard2.sav')


            X = modeloS.transform(VectorCarac)
            
            X_test = np.array(X).reshape(-1,1)

            prediccion=modeloSVM.predict(X_test)

            if(prediccion == 0):
                print ("Canino")
            elif(prediccion == 1):
                print("Incisivo")
            else:
                print("Molar")
                
            cv2.imshow('image',img)
            cv2.waitKey(0)
        else:
            cv2.drawContours(img,[currentContour],0,(0,0,255),3)
            print ("Scrap")
            cv2.imshow('image',img)
            cv2.waitKey(0)

cv2.destroyAllWindows()
