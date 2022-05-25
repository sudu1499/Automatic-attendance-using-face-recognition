import tensorflow as tf
from utils.model import get_model
from utils.get_x_y import get_x_y
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import yaml
from utils.save_croped import save_crp

from utils.cusotm_model import custom_model

config=yaml.safe_load(open('utils\config.yaml','r'))
model_path=config['model_path']

################### for new students ############################
#save_crp(config)

#################################################################

x_train,x_test,y_train,y_test=get_x_y()
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=.25)

(model,summary)=get_model(config)

###################################for custom network#########################
#model=custom_model(config)
##############################################################################

clb=EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True)

model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=10,callbacks=[clb],batch_size=64)

model.save(model_path)
