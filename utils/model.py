import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
import io

def get_model(config):
    xcep=Xception(include_top=False,weights='imagenet',input_shape=(config['size'],config['size'],3))
    for i in xcep.layers:
        i.trainable=False###try to trian laste layer
    model=tf.keras.models.Sequential()
    model.add(xcep)
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dropout(rate=.3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=500,activation='relu',kernel_initializer='HeNormal'))#kernel_regularizer=tf.keras.regularizers.l2(.001))

    model.add(tf.keras.layers.Dropout(rate=.3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=500,activation='relu',kernel_initializer='HeNormal'))

    model.add(tf.keras.layers.Dropout(rate=.3))
    model.add(tf.keras.layers.BatchNormalization())    
    model.add(tf.keras.layers.Dense(units=500,activation='relu',kernel_initializer='HeNormal'))

    model.add(tf.keras.layers.Dropout(rate=.3))
    model.add(tf.keras.layers.BatchNormalization())    
    model.add(tf.keras.layers.Dense(units=100,activation='relu',kernel_initializer='HeNormal'))

    model.add(tf.keras.layers.Dropout(rate=.3))
    model.add(tf.keras.layers.BatchNormalization())    
    model.add(tf.keras.layers.Dense(units=config['no_students'],activation='softmax'))

    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    def log(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x:stream.write(f'{x}\n'))
            summary=stream.getvalue()
            return(summary)
    summary=log(model)
    
    return (model,summary)
if __name__=='__main__':
    get_model({})
##100,50,50 is good
##100,100,50 is more better