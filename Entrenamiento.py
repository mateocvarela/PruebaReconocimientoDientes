# -*- coding: cp936 -*-
import numpy as np
import cv2
import xlsxwriter
row = 0
col = 0
j=1
i=1

workbook = xlsxwriter.Workbook('DatosNumeros3.xlsx')
worksheet = workbook.add_worksheet('Numeros')

img = cv2.imread("molares.tif")
cv2.namedWindow('Original',cv2.WINDOW_NORMAL)
cv2.imshow('Original',img)
cv2.waitKey(0)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray,5)
ret,edges = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



drawing = np.zeros(img.shape,np.uint8)

contours,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)        
cv2.drawContours(img, contours,-1 , (0,255,0), 3)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
hierarchy = hierarchy[0]


for component in zip(contours, hierarchy):
                
            cnt = component[0]
            currentHierarchy = component[1]
            
            if (cv2.contourArea(cnt)>0 and cv2.contourArea(cnt)< 100000):
                    x,y,w,h = cv2.boundingRect(cnt)
                    if (currentHierarchy[1] < 20):
                        # Contornos más externos  o padres, no tiene en cuenta los agujeros
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
                        print Hu[0][0], Hu[1][0], Hu[2][0], Hu[3][0], Hu[4][0], Hu[5][0],Hu[6][0], "\n"
                        VectorCarac = np.array([A, p, Comp, Hu[0][0], Hu[1][0], Hu[2][0], Hu[4][0]], dtype = np.float32)
                        # Se itera el vector de características y se escribe en el archivo
                        # Se conserva
                        for carac in (VectorCarac):
                            worksheet.write(row, col, j)
                            worksheet.write(row, i, carac)
                            i=i+1
                        i=1
                        row += 1
                        cv2.namedWindow('img',cv2.WINDOW_NORMAL)
                        


                        cv2.imshow('img',img)
                        cv2.waitKey(0)
                        

                
# Se cierra el archivo
workbook.close()
    
# Se muestra la imagen y se cierran las ventanas
#cv2.imshow('img',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
