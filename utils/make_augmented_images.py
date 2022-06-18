from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import numpy as np
import cv2


def create_aug_images(config): #does augmentation for all images in the same folder
    path=config['img_path']
    size=config['size']
    datagen=ImageDataGenerator(width_shift_range=.1,height_shift_range=.1,zoom_range=.2,rotation_range=45,fill_mode='constant',cval=125)

    for i in glob(path+'\*'):
        name=i.split('\\')[-1]
        for j in glob(i+'\*'):
            img=cv2.imread(j,1)
            img=cv2.resize(img,(size,size))
            img=np.reshape(img,((1,)+img.shape))
            c=0
            for d in datagen.flow(img,batch_size=1,save_to_dir=path+'\\'+name+'\\',save_format='.jpeg'):
                c+=1
                if c==5:
                    break

