from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import cv2
import yaml

config=yaml.safe_load(open('utils\config.yaml','r'))
path=config['img_path']+'\\*'
size=config['size']
datagen=ImageDataGenerator(rotation_range=.5,shear_range=.3,zoom_range=.4,width_shift_range=.2,height_shift_range=.3,fill_mode='nearest',brightness_range=(.1,.9))           

for i in glob(path):
    name=i.split('\\')[-1]
    x=[]
    for j in glob(i+'\\*'):
        print(j)
        t=cv2.imread(j)
        t=cv2.resize(t,(size,size))
        t=t.reshape((1,size,size,3))
        x.append(t)
    for xi in x:
        i=0
        for _ in datagen.flow(xi,save_to_dir=f'.\\images\\{name}\\',save_format='.jpg'):
            i+=1
            if i==30:
                break