from __future__ import print_function, division
import numpy as np
import cv2
import xlrd
from time import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
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
        if(sh.cell_value(rowx=i, colx=0)==1):
            y.append(0)
        elif(sh.cell_value(rowx=i, colx=0)==2):
            y.append(1)
        elif(sh.cell_value(rowx=i, colx=0)==3):
            y.append(2)
    y= np.array(y,np.float32)
    return x,y

if __name__ == '__main__':
    t0 = time()
    X, Y = load_xlsx(book_excel)
    
    standard_scaler = StandardScaler()
    X_S = standard_scaler.fit_transform(X)
        
    robust_scaler = RobustScaler()
    X_R = robust_scaler.fit_transform(X)

    samples_train, samples_test, responses_train, responses_test = \
                train_test_split(X_S, Y, test_size = 0.3)
    samples_train1, samples_test1, responses_train1, responses_test1 = \
                train_test_split(X_R, Y, test_size = 0.3)
        
    svm =SVC(C=100, kernel="rbf")
    svm1 =SVC(C=100, kernel="rbf")

    svm.fit(samples_train,responses_train)
    svm1.fit(samples_train1,responses_train1)

    response_pred = svm.predict(samples_test)
    response1_pred = svm1.predict(samples_test1)

    scoresS = cross_val_score(svm, X_S,Y, cv=10, scoring='accuracy')
    scoresR = cross_val_score(svm1, X_R,Y, cv=10, scoring='accuracy')

    response_pred = svm.predict(samples_test)
    response_pred1 = svm1.predict(samples_test1)

    print (scoresS.mean())
    print (scoresR.mean())

if (scoresS.mean()>scoresR.mean()):
    fileST='ModeloStandard2.sav'
    joblib.dump(standard_scaler,fileST)
    filename='ModeloSVM_S2.sav'
    joblib.dump(svm,filename)
    print ("done in %0.16fs" % (time() - t0))
else:
    fileST1='ModeloRobust2.sav'
    joblib.dump(robust_scaler,fileST1)
    filename='ModeloSVM_R2.sav'
    joblib.dump(svm1,filename)
    print ("done in %0.16fs" % (time() - t0))
