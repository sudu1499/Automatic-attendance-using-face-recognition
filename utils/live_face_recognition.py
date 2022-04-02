import cv2
import dlib
import numpy as np
import yaml
from tensorflow.keras.models import load_model
import pickle
config=yaml.safe_load(open('utils\config.yaml','r'))
path=config['img_path']
model_path=config['model_path']
size=config['size']
encoder=pickle.load(open(config['encoder_path'],'rb'))

model=load_model(model_path)#################### or use live model from main.py##############
det=dlib.get_frontal_face_detector()
vid=cv2.VideoCapture(0)
while 1:
    _,frame=vid.read()
    frame_g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    d=det(frame_g)
    if len(d):
        for i in d:
            crp=frame[i.top():i.bottom(),i.left():i.right()]
            crp=cv2.resize(crp,(size,size))
            cv2.imshow('c',crp)
            crp=np.reshape(crp,(1,size,size,3))
            print( encoder.inverse_transform([[1 if i>=.5 else 0 for i in model.predict(crp)[0] ] ] ))
            #print(model.predict(crp))
        if cv2.waitKey(1)==ord('q'):
            break
cv2.destroyAllWindows()
vid.release()