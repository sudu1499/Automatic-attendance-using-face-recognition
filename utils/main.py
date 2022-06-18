from pydoc import describe
from cv2 import destroyAllWindows
import tensorflow as tf
from utils.live_face_recognition import live_recognize
from utils.model import get_model
from utils.get_x_y import get_x_y, get_x_y_from_folder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import yaml
from utils.save_croped import save_crp
from utils.make_augmented_images import create_aug_images
from utils.cusotm_model import custom_model
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import cv2
config=yaml.safe_load(open('utils\config.yaml','r'))
model_path=config['model_path']

################### for new students ############################
#save_crp(config)

#################################################################

######################for augmentaion #############################
#create_aug_images(config)
###################################################################

x_train,x_test,y_train,y_test=get_x_y_from_folder()
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=.25)

(model,summary)=get_model(config)


clb=EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True)

model.fit(x_train/255,y_train,validation_data=(x_val/255,y_val),epochs=12,callbacks=[clb],batch_size=64)

live_recognize(model)

model.save(model_path)
