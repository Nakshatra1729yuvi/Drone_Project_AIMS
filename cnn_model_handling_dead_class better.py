import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import mediapipe as mp
import os
import joblib
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
import warnings
warnings.filterwarnings("ignore")


path_model='D:\\Drone_AIMS_Scratch\\cnn.pth'
scaler_path='D:\\Drone_AIMS_Scratch\\scaler_cnn.pkl'

dict_map={'Thumbs_Up':0,'Thumbs_Down':1,'Index_Point_Up':2,'Index_Point_Down':3,'Open_Hand':4,'Closed_Fist':5,'Victory':6,'Yo':7,'Little_Finger_Up':8,'Thumb_Left':9,'Thumb_Right':10,'Dead':11}
dict_map_inv={0:'Thumbs_Up',1:'Thumbs_Down',2:'Index_Point_Up',3:'Index_Point_Down',4:'Open_Hand',5:'Closed_Fist',6:'Victory',7:'Yo',8:'Little_Finger_Up',9:'Thumb_Left',10:'Thumb_Right',11:'Dead'}
label_map={"Thumbs_Up":"Up","Thumbs_Down":"Down","Index_Point_Up":"Up","Index_Point_Down":"Down","Open_Hand":"Forward","Closed_Fist":"Backward","Victory":"Landing","Yo":"BackFlip","Little_Finger_Up":"None","Thumb_Left":"Left","Thumb_Right":"Right","Dead":"None"}
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GestureModel(nn.Module):
    def __init__(self,input_size,num_classes):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(in_channels=3,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128,out_channels=256,kernel_size=6,stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier=nn.Linear(256,num_classes)
    
    def forward(self,x):
        x=x.view(x.shape[0],3,-1)
        x=self.net(x)
        return self.classifier(x)

gModel=GestureModel(63,12).to(device)

if os.path.exists(path_model):
    gModel.load_state_dict(torch.load(path_model))
    scaler=joblib.load(scaler_path)
else:
    df=pd.read_csv('D:\\Drone_AIMS_Scratch\\data_aug.csv')

    X=df.iloc[:,1:-1].to_numpy(dtype=np.float32)
    y=df.iloc[:,-1].astype(str)
    

    y=np.array([dict_map[i] for i in y])

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)


    X_train=torch.tensor(X_train,dtype=torch.float32)
    X_test=torch.tensor(X_test,dtype=torch.float32)
    y_train=torch.tensor(y_train,dtype=torch.long)
    y_test=torch.tensor(y_test,dtype=torch.long)
    
    train_dataset=DataLoader(TensorDataset(X_train,y_train),batch_size=32,shuffle=True)
    test_dataset=DataLoader(TensorDataset(X_test,y_test),batch_size=32,shuffle=False)

    optimizer=optim.Adam(gModel.parameters(),lr=0.001)
    loss_fn=nn.CrossEntropyLoss()
    epochs=20
    for epoch in tqdm(range(epochs)):
        for x_t,y_t in train_dataset:
            x_t=x_t.to(device)
            y_t=y_t.to(device)            
            y_pred=gModel(x_t)
            loss=loss_fn(y_pred,y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        logits=gModel(X_test.to(device))
        y_pred=torch.argmax(logits,dim=1)
        total=0
        correct=0
        for i in range(len(y_test)):
            if y_test[i]==y_pred[i]:
                correct+=1
            total+=1
        print(f"Accuracy: {correct*100/total}")

    #saving model
    torch.save(gModel.state_dict(),path_model)
    joblib.dump(scaler,scaler_path)


cap=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
mp_drawing=mp.solutions.drawing_utils

gModel.eval()
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
        data_pt=scaler.transform(data_pt)
        logits=gModel(torch.tensor(data_pt,dtype=torch.float32).to(device))
        prediction=torch.argmax(logits).item()
        confidence=torch.softmax(logits,dim=1)[0][prediction].item()
        if confidence<0.7:
            prediction=11
        cv2.putText(frame,label_map[dict_map_inv[prediction]],(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



