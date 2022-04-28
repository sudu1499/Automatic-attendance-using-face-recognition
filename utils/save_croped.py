import cv2
import dlib
import os
import yaml

def save_crp(config):
    count=0
    path='.\\images'
    vid=cv2.VideoCapture(0)
    det=dlib.get_frontal_face_detector()
    no_students=config['no_students']
    name=input('ur name')
    try:
        os.makedirs(path+f'\\{name}',exist_ok=True)
    except:
        print('folder seems to be already exist',path+f'\\{name}')
        return None
    no_students+=1
    config['no_students']=no_students
    yaml.dump(config,open('E:\\project\\new_face_recognition\\utils\\config.yaml','w'))
    while 1:
        _,frame=vid.read()
        gframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        d=det(gframe)
        if d and count!=101:
            for i in d :
                crp=frame[i.top():i.bottom(),i.left():i.right()]
                crp=cv2.resize(crp,(240,240))
                cv2.imshow('crp',crp)
                cv2.imwrite(path+f'\\{name}\\{count}.jpg',crp)
            count+=1
        if cv2.waitKey(1)==ord('q'):
            break
        
    cv2.destroyAllWindows()
    vid.release()
if __name__=='__main__':
    save_crp({})

