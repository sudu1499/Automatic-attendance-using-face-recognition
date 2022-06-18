import sqlite3
import cv2
import dlib
import numpy as np
import yaml
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime

def live_recognize():
    config=yaml.safe_load(open('utils\config.yaml','r'))
    path=config['img_path']
    model_path=config['model_path']
    size=config['size']
    encoder=pickle.load(open(config['encoder_path'],'rb'))
    no_students=config['no_students']

    model=load_model(model_path)#################### or use live model from main.py##############
    det=dlib.get_frontal_face_detector()
    conn=sqlite3.connect('attendance.db')
    c=conn.cursor()
    vid=cv2.VideoCapture(0)
    d=datetime.now()
    c.execute(f'select * from student where date="{str(d.year)+"-"+str(d.month)+"-"+str(d.day)}"')
    students=[]
    for i in c.fetchall():
        students.append(i[0])
    name=''
    while 1:
        _,frame=vid.read()
        frame_g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        d=det(frame_g)
        if len(d):
            t=list()
            for i in d:
                t=frame[i.top():i.bottom(),i.left():i.right()]
                t_copy=t

            t=cv2.resize(t,(size,size))
                    # crp=np.reshape(crp,(1,size,size,3))
                    # print( encoder.inverse_transform([[1 if i>=.5 else 0 for i in model.predict(crp)[0]]]))
                    #print(np.argmax(model.predict(crp)))
                #cv2.imshow(f'{i}',)
            t=np.reshape(t,(1,size,size,3))
                    #k.append(np.argmax(model.predict(j)))
                    #_=cv2.putText(m,f'{encoder.inverse_transform([[1 if i>=.5 else 0 for i in model.predict(j)[0]]])[0] }',org=(0,150), fontFace=cv2.FONT_HERSHEY_TRIPLEX,lineType=cv2.LINE_AA,thickness=3,color=(0,0,255),fontScale=1)
            h=np.zeros((1,no_students))
            h[0][np.argmax(model.predict(t/255))]=1
            name=encoder.inverse_transform(h)[0][0]
            if name not in students:
                d=datetime.now()
                q=f'insert into student values("{name}","{str(d.year)+"-"+str(d.month)+"-"+str(d.day)}")'
                print(q)
                c.execute(q)
                students.append(name)
            _=cv2.putText(t_copy,f'{encoder.inverse_transform(h)[0][0] }',org=(0,150), fontFace=cv2.FONT_HERSHEY_TRIPLEX,lineType=cv2.LINE_AA,thickness=3,color=(0,0,255),fontScale=1)
            cv2.imshow('recognized',t_copy)
                
            if cv2.waitKey(1)==ord('q'):
                conn.commit()
                break
        else:
            try:
                cv2.destroyAllWindows()
            except:
                continue
    cv2.destroyAllWindows()
    vid.release()

live_recognize()