# -*- coding: cp936 -*-
import numpy as np
import cv2
import xlrd
from time import time

#Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
#import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

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

##### Inicio del programa ######
if __name__ == '__main__':
    t0 = time()
   
    
    # Cargar datos desde un archivo .xlsx
    # la función retornará el número de muestras obtenidas y su respectiva clase
    X, Y = load_xlsx(book_excel)

    # Se separan los datos: un % para el entrenamiento del modelo y otro
    # para el test
    samples_train, samples_test, responses_train, responses_test = \
    train_test_split(X, Y, test_size = 0.1)

    
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=1)
    
    knn.fit(samples_train, responses_train)
    response_pred = knn.predict(samples_test)

    print "accuracy_score: ", knn.score(samples_test, responses_test)
    print "accuracy_score: ", accuracy_score(responses_test, response_pred)
    print "\n"
    print "Matriz de confusion: \n",confusion_matrix(responses_test, response_pred)
    print("done in %0.16fs" % (time() - t0))

    scores = cross_val_score(knn, X,Y, cv=10, scoring='accuracy')
    print scores
    print scores.mean()

    k_range = range(1,31)
    k_scores=[]
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=1)
        scores = cross_val_score(knn, X,Y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    print k_scores

    """plt.plot(k_range,k_scores)
    plt.xlabel('Value of k for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()"""
