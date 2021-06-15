import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def checkMask():
    with_mask=np.load('with_mask.npy') #[200,50,50,3]
    without_mask=np.load('without_mask.npy') #[200,50,50,3]

    with_mask=with_mask.reshape(200,50*50*3)
    without_mask=without_mask.reshape(200,50*50*3)
    # print(without_mask.shape)
    # print(with_mask.shape)

    #combining both data
    X=np.r_[with_mask,without_mask] 



    #first 200 => with_mask =>0
    #last 200 => without_mask => 1
    labels=np.zeros(X.shape[0])
    labels[200:]=1.0 
    names={0:'Mask',1:'No mask'}

    #spliting data for testing and traing purpose
    x_train,x_test,y_train,y_test=train_test_split(X,labels,test_size=0.25)

    #dimensionally reduction
    pca=PCA(n_components=3)
    x_train=pca.fit_transform(x_train)
    print(x_train.shape)



    # #machine learning
    svm=SVC()
    svm.fit(x_train,y_train)
    print(svm)


    x_test = pca.transform(x_test)
    y_pred = svm.predict(x_test)
    print(accuracy_score(y_test,y_pred))



    # x_test=pca.transform(x_test)
    # y_pred=svm.predict(x_test)

    # print(accuracy_score(y_test,y_pred))

    # #start video
    haar_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    capture=cv2.VideoCapture(0)
    data=[]
    font=cv2.FONT_HERSHEY_COMPLEX
    while True:
        flag,img=capture.read()
        if flag:
            faces=haar_data.detectMultiScale(img)
            for x,y,w,h in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)
                face=img[y:y+h,x:x+w,:]
                face=cv2.resize(face,(50,50))
                face=face.reshape(1,-1)
                face=pca.transform(face)
                pred=svm.predict(face)
                n=names[int(pred)]
                cv2.putText(img,n,(x,y),font,1,(255,0,255),2)
                print(n)
            cv2.imshow('result',img)
            if cv2.waitKey(2)==27 :
                break
    capture.release()