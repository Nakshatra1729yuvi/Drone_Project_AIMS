import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import cv2
import mediapipe as mp
import os
import joblib
import warnings
warnings.filterwarnings("ignore")


path_model='D:\\Drone_AIMS_Scratch\\model.joblib'

dict_map={'Thumbs_Up':0,'Thumbs_Down':1,'Index_Point_Up':2,'Index_Point_Down':3,'Open_Hand':4,'Closed_Fist':5,'Victory':6,'Yo':7,'Little_Finger_Up':8,'Thumb_Left':9,'Thumb_Right':10,'Dead':11}
dict_map_inv={0:'Thumbs_Up',1:'Thumbs_Down',2:'Index_Point_Up',3:'Index_Point_Down',4:'Open_Hand',5:'Closed_Fist',6:'Victory',7:'Yo',8:'Little_Finger_Up',9:'Thumb_Left',10:'Thumb_Right',11:'Dead'}

if os.path.exists(path_model):
    model=joblib.load(path_model)
else:
    df=pd.read_csv('D:\\Drone_AIMS_Scratch\\data_aug.csv')

    X=df.values[:,1:-1]
    y=df.values[:,-1]
    

    y=np.array([dict_map[i] for i in y])

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


    model=RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    print(accuracy_score(y_test,y_pred))

    #saving model
    import joblib
    joblib.dump(model,'D:\\Drone_AIMS_Scratch\\model.joblib')


cap=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
mp_drawing=mp.solutions.drawing_utils

while True:
    ret,frame=cap.read()
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)
    if result.multi_hand_landmarks:
        hand=result.multi_hand_landmarks[0]
        x_cord=[lm.x for lm in hand.landmark]
        y_cord=[lm.y for lm in hand.landmark]
        z_cord=[lm.z for lm in hand.landmark]
        data_pt=np.array(x_cord+y_cord+z_cord)
        data_pt=data_pt.reshape(1,-1)
        prediction=model.predict(data_pt)
        cv2.putText(frame,dict_map_inv[prediction[0]],(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



