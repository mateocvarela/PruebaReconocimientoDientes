# -*- coding: cp936 -*-

import numpy as np
import cv2
import xlrd
from time import time

#Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import accuracy_score #Dice que tan bueno y malo es el modelo
from sklearn.metrics import confusion_matrix #Permite ver en que se equivoca más el modelo

from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

book_excel = xlrd.open_workbook("DatosNumeros1.xlsx")


def load_xlsx(xlsx):
    
    sh = xlsx.sheet_by_index(0)
    x = np.zeros((sh.nrows,sh.ncols-1))
    y = []
    for i in range(0, sh.nrows):
        for j in range(0, sh.ncols-1):
            x[i,j] = sh.cell_value(rowx=i, colx=j+1)
            
        if(sh.cell_value(rowx=i, colx=0)==0):
            y.append(0)
        elif(sh.cell_value(rowx=i, colx=0)==1):
            y.append(1)
        elif(sh.cell_value(rowx=i, colx=0)==2):
            y.append(2)
        elif(sh.cell_value(rowx=i, colx=0)==3):
            y.append(3)
        elif(sh.cell_value(rowx=i, colx=0)==4):
            y.append(4)
        elif(sh.cell_value(rowx=i, colx=0)==5):
            y.append(5)
        elif(sh.cell_value(rowx=i, colx=0)==6):
            y.append(6)
        elif(sh.cell_value(rowx=i, colx=0)==7):
            y.append(7)
        elif(sh.cell_value(rowx=i, colx=0)==8):
            y.append(8)
        elif(sh.cell_value(rowx=i, colx=0)==9):
            y.append(9)
    y= np.array(y,np.float32)
    return x,y
ScoreS = 0
ScoreR = 0
##### Inicio del programa ######
##while True:
if __name__ == '__main__':
        t0 = time()
    
        # Cargar datos desde un archivo .xlsx
        # la función retornará el número de muestras obtenidas y su respectiva clase
        X, Y = load_xlsx(book_excel)
        standard_scaler = StandardScaler()
        X_S = standard_scaler.fit_transform(X)
        robust_scaler = RobustScaler()
        X_R = robust_scaler.fit_transform(X)

        #print salidas
        #print Y
    
        # Se separan los datos: un % para el entrenamiento del modelo y otro
        # para el test
        samples_train, samples_test, responses_train, responses_test = \
                    train_test_split(X_S, Y, test_size = 0.3)

        samples_train1, samples_test1, responses_train1, responses_test1 = \
                    train_test_split(X_R, Y, test_size = 0.3)

   
        mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(10,10), max_iter=1000, tol=0.0001)
        mlp1 = MLPClassifier(activation='tanh', hidden_layer_sizes=(10,10), max_iter=1000, tol=0.0001)
    
        mlp.fit(samples_train, responses_train)
        mlp1.fit(samples_train1, responses_train1)
    
        response_pred = mlp.predict(samples_test) #Saca las salidas (0,1,2)
        response_pred1 = mlp1.predict(samples_test1)

        print "Standard Scaler"
        ScoreS = cross_val_score(mlp, X_S,Y, cv=5, scoring='accuracy')
        print ScoreS[0]
        
        print "Robust Scales"
        ScoreR = cross_val_score(mlp1, X_R,Y, cv=5, scoring='accuracy')
        print ScoreR[0]
        print ("\n")
##        print "Standard Scaler"
##        print "Accuracy_score: ", mlp.score(samples_test, responses_test)*100
##        print "Accuracy_score: ", accuracy_score(responses_test, response_pred)*100
##        print ("\n")
##
##        print "Robust Scales"
##        print "Accuracy_score: ", mlp1.score(samples_test1, responses_test1)*100
##        print "Accuracy_score: ", accuracy_score(responses_test1, response_pred1)*100
##        print ("\n")
##        print("done in %0.16fs" % (time() - t0))
##        print ("\n")
##        ScoreS = mlp.score(samples_test, responses_test)*100
##        ScoreR = mlp1.score(samples_test1, responses_test1)*100
##        svm_matrix = confusion_matrix(responses_test,response_pred)
##        print (svm_matrix)
        if (ScoreS[0] >= 0.85 and ScoreS[0] > ScoreR[0]):
            svm_matrix = confusion_matrix(responses_test,response_pred)
            print (svm_matrix)
            joblib.dump(standard_scaler,"Modelo_Standard.sav")
            joblib.dump(mlp, "ModeloMlp_S.sav")
            print "Se guardo modelo Standard Scaler con exactitud de: ", ScoreS
##            break
        
        if (ScoreS[0] >= 0.85 and ScoreS[0] == ScoreR[0]):
            svm_matrix = confusion_matrix(responses_test,response_pred)
            print "Matriz Standard", (svm_matrix)
            svm_matrix1 = confusion_matrix(responses_test1,response_pred1)
            print "Matriz Robust", (svm_matrix1)
            joblib.dump(standard_scaler,"Modelo_Standard.sav")
            joblib.dump(mlp, "ModeloMlp_S.sav")
            joblib.dump(robust_scaler,"Modelo_Robust.sav")
            joblib.dump(mlp, "ModeloMlp_R.sav")
            print "Se guardo modelo Standard Scaler con exactitud de: ", ScoreS
            print "Se guardo modelo Robust Scaler con exactitud de: ", ScoreR
##            break
        if ScoreR[0] >= 0.85 and ScoreR[0] > ScoreS[0]:
            svm_matrix1 = confusion_matrix(responses_test1,response_pred1)
            print (svm_matrix1)
            joblib.dump(robust_scaler,"Modelo_Robust.sav")
            joblib.dump(mlp, "ModeloMlp_R.sav")
            print "Se guardo modelo Robust Scaler con exactitud de: ", ScoreR
##            break



