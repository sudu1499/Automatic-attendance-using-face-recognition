import cv2
import numpy as np
from glob import glob
def resize(path,size):
    for i in glob(path+'\\*'):
        name=i.split('\\')[-1]
        print(name)
        for j in glob(i+'\\*'):
            img=cv2.imread(j)
            img=cv2.resize(img,(size,size))
            cv2.imwrite(j,img)
resize('images',160)