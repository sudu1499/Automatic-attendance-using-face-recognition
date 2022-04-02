import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
def get_model(config):
    xcep=Xception(include_top=False,weights='imagenet',input_shape=(config['size'],config['size'],3))
    for i in xcep.layers:
        i.trainable=False
    model=tf.keras.models.Sequential()
    model.add(xcep)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=100,activation='relu'))
    model.add(tf.keras.layers.Dense(units=100,activation='relu'))
    model.add(tf.keras.layers.Dense(units=50,activation='relu'))
    model.add(tf.keras.layers.Dense(units=config['no_students'],activation='softmax'))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    print(model.summary())
    return model
m=get_model()


##100,50,50 is good
##100,100,50 is more better