import cv2
import dlib
import numpy as np
import yaml
from tensorflow.keras.models import load_model
config=yaml.safe_load(open('utils\config.yaml','r'))
path=config['img_path']
model_path=config['model_path']
size=config['size']
det=dlib.get_frontal_face_detector()
vid=cv2.VideoCapture(0)
model=load_model(model_path)#################### or use live model from main.py##############
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
            print(np.argmax(model.predict(crp)))
        if cv2.waitKey(1)==ord('q'):
            break
cv2.destroyAllWindows()
vid.release()