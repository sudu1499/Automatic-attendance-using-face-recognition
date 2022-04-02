import numpy as np
from glob import glob
import yaml
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import yaml
def get_x_y():
    config=yaml.safe_load(open('utils\config.yaml','r'))
    path=config['img_path']
    size=config['size']    
    ohe=OneHotEncoder()
    y=[]
    x=[]
    for i in glob(path+'\\*'):
        name=i.split('\\')[-1]
        for j in glob(i+'\\*'):
            img=cv2.imread(j)
            img=cv2.resize(img,(size,size))################################################        resize ###############
            x.append(img)
            y.append(name)
    x=np.array(x)
    yy=np.array(y)
    y=ohe.fit_transform(yy.reshape((-1,1))).toarray()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
    return x_train,x_test,y_train,y_test
    get_x_y()