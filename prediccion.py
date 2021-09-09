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

modelo=joblib.load('ModeloMlp_R.pkl')
modelost=joblib.load('Modelo_Robust.pkl')
print (modelost)
print (modelo)
arduinoPort= serial.Serial('COM5',9600 , timeout=1)

captura= cv2.VideoCapture(1)
time.sleep(2)

while (captura.isOpened()):
    
    ret, video = captura.read()
    if ret == True:
        cv2.imshow("Video" ,video)
        wv,hv = video.shape[:2]
        #print (wv,hv)
        #video = cv2.imread('1/Diente_10.jpg')
        
        kernel = np.ones((3,3),np.uint8)
        c=0
        
    
        gray = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray,5)
        ret,edges = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
        drawing = np.zeros(video.shape,np.uint8)
        # Image to draw the contours

        # Encontrar contornos
        try:
            
            contours,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            hierarchy = hierarchy[0]
            #cv2.waitKey(1)
     
            for component in zip(contours, hierarchy):
                
                cnt = component[0]
                currentHierarchy = component[1]
                #print (cv2.contourArea(cnt))
                if(cv2.contourArea(cnt)>4000):
                    captura.release()
                    cv2.destroyAllWindows()
                    #time.sleep(1)
                    captura= cv2.VideoCapture(1)
                    time.sleep(2)
                    ret2, video2 = captura.read()
                    gray2 = cv2.cvtColor(video2,cv2.COLOR_BGR2GRAY)
                    blur2 = cv2.medianBlur(gray2,5)
                    ret2,edges2 = cv2.threshold(blur2,50,255,cv2.THRESH_BINARY)
                    drawing = np.zeros(video2.shape,np.uint8)
                    
                    contours,hierarchy = cv2.findContours(edges2.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    hierarchy = hierarchy[0]

                    for component in zip(contours, hierarchy):
                
                        cnt = component[0]
                        currentHierarchy = component[1]
                    
                    
                    
                    
                    x,y,w,h = cv2.boundingRect(cnt)
                    if (currentHierarchy[3] < 0):
                        cv2.drawContours(edges2,[cnt],0,(120,150,20),1)
                        cv2.imshow("gradiente",edges2)
                        
                        cv2.waitKey(1)
                        
                        cv2.rectangle(video2,(x,y),(x+w,y+h),(0,0,255),1)
                                    
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
                        VectorCarac = np.array([p, Comp,RA,Hu[0][0], Hu[1][0], Hu[2][0], Hu[3][0], Hu[4][0], Hu[5][0],Hu[6][0]], dtype = np.float32)
                                    

                
                        VectorCarac= modelost.transform(VectorCarac)
                        VectorCarac_test=np.array(VectorCarac).reshape((1,-1))


                        Prediccion=modelo.predict(VectorCarac_test)

                                
                        print (Prediccion)
                        if Prediccion == 0:
                            
                            
                            flag= "A"
                            time.sleep(1)
                            arduinoPort.write(flag)
                            time.sleep(2)
                            
                        if Prediccion == 1:
                            
                            flag= "B"
                            time.sleep(1)
                            arduinoPort.write(flag)
                            time.sleep(2)
                            
                        if Prediccion == 2:
                            
                            
                            flag= "C"
                            time.sleep(1)
                            arduinoPort.write(flag)
                            time.sleep(2)
                        #time.sleep(2)

        except:
            pass
                        
                        
            
    else:
        break
    if (cv2.waitKey(1) & 0xFF ==ord('q')):
        flag= "A"
        time.sleep(1)
        arduinoPort.write(flag)
        time.sleep(2)
        arduinoPort.close()
        break
captura.release()
cv2.destroyAllWindows()
    

    





