from __future__ import print_function, division
import xlsxwriter
import numpy as np
import cv2
from glob import glob
import time
import serial


import xlrd
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score

modelo=joblib.load('ModeloMlp_R.sav')
modelost=joblib.load('Modelo_Robust.sav')
print (modelost)
print (modelo)

img = cv2.imread("mixtos2.tif")
cv2.namedWindow('Original',cv2.WINDOW_NORMAL)
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
kernel = np.ones((3,3),np.uint8)
c=0


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray,5)
ret,edges = cv2.threshold(blur,70,255,cv2.THRESH_BINARY)
drawing = np.zeros(img.shape,np.uint8)
# Image to draw the contours

    
contours,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours,-1 , (0,255,0), 3)

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
hierarchy = hierarchy[0]
    #cv2.waitKey(1)

for component in zip(contours, hierarchy):
    
    cnt = component[0]
    currentHierarchy = component[1]
    #print (cv2.contourArea(cnt))
    if (cv2.contourArea(cnt)>5000 and cv2.contourArea(cnt)< 15000):
        x,y,w,h = cv2.boundingRect(cnt)            
        
        if (currentHierarchy[1] < 255):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            
               
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            A = cv2.contourArea(cnt)
            p = cv2.arcLength(cnt,True)
            Comp = A/float(p*p)
            RA = w/float(h)
                        #print cx,cy,A,p,Comp,RA
                
            Hu = cv2.HuMoments(M) #Normaliza los 32 momentos y solo arroja 7
                        #print Hu
            #print Hu[0][0], Hu[1][0], Hu[2][0], Hu[3][0], Hu[4][0], Hu[5][0],Hu[6][0], "\n"
            VectorCarac = np.array([A, p, Comp, Hu[0][0], Hu[1][0], Hu[2][0], Hu[4][0]], dtype = np.float32)
                        

    
            VectorCarac= modelost.transform(VectorCarac)
            VectorCarac_test=np.array(VectorCarac).reshape((1,-1))


            Prediccion=modelo.predict(VectorCarac_test)

           
            cv2.namedWindow('gradientes',cv2.WINDOW_NORMAL)
            
            cv2.imshow("gradientes",img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
           

            print (Prediccion)

            if (Prediccion == 1):
                print ("este diente es un canino")
            elif Prediccion == 2:
                print ("este diente es un insicivo")
            elif Prediccion == 3:
                print ("este diente es un molar")
