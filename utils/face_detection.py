import cv2
import dlib 
from glob import glob
vid=cv2.VideoCapture(0)
det=dlib.get_frontal_face_detector()
while 1:
    _,frame=vid.read()
    frame_g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    d=det(frame_g)
    if len(d):
        for i in d:
            # crp=frame[i.top():i.bottom(),i.left():i.right()]
            # crp=cv2.resize(crp,(400,400))
            crp=cv2.rectangle(frame,(i.left(),i.top()) ,(i.right(),i.bottom()),color=(255,0,0),thickness=2)
        cv2.imshow('',crp)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
vid.release() 


#only for yash boss
# def detect_from_folder(path):
#     j=0
#     for i in glob(path):
#         img=cv2.imread(i)
#         imgg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         d=det(imgg)
#         if len(d):
#             for i in d:
#                 crp=img[i.top():i.bottom(),i.left():i.right()]
#                 cv2.imwrite(f'.\\images3\\yash\\{j}.jpg',crp)
#                 j+=1
# detect_from_folder('.\\images\\yash\\y\\*')