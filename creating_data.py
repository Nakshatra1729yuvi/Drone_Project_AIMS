import cv2
import mediapipe as mp
import csv

# #Index_Finger_up=UP
# #Index_Finger_down=DOWN
# #Thumb_left=LEFT
# #Thumb_right=RIGHT
# #Open_hand=forward
# #Fist=backward
# #Yo_Yo=BackFLip
# #Victory=Landing
# #Dead=Dead


#Thumbs_Up
#Thumbs_Down
#Index_Point_Up
#Index_Point_Down
#Open_Hand
#Closed_Fist
#Victory
#Yo
#Little_Finger_Up
#Thumb_Left
#Thumb_Right
#Dead

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(max_num_hands=1)

cap=cv2.VideoCapture(0)
buffer=[]

gesture_label=input("Enter gesture label: ")
no_of_samples=int(input("Enter no of samples: "))
sample=0
while sample<no_of_samples:
    ret,frame=cap.read()
    cv2.imshow("Frame",frame)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)
    if result.multi_hand_landmarks:
        sample+=1
        hand=result.multi_hand_landmarks[0]
        x_cord=[lm.x for lm in hand.landmark]
        y_cord=[lm.y for lm in hand.landmark]
        z_cord=[lm.z for lm in hand.landmark]
        data_pt=[gesture_label]+x_cord+y_cord+z_cord
        buffer.append(data_pt)
        print('taken')
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    
with open("D:\\Drone_AIMS_Scratch\\data.csv","a") as f:
    csv_writer=csv.writer(f)
    csv_writer.writerows(buffer)


cap.release()
cv2.destroyAllWindows()
    
