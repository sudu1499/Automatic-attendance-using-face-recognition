from keras.layers import Dropout,Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dense
from keras.models import Sequential

def model2(config):

    model=Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),padding='valid',input_shape=(160,160,3),activation='relu'))#158*158*32
    model.add(BatchNormalization())
    model.add(Dropout(rate=.2))

    model.add(MaxPooling2D(pool_size=(2,2)))#79*79*32

    model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu'))#77*77*32
    model.add(BatchNormalization())
    model.add(Dropout(rate=.2))

    model.add(MaxPooling2D(pool_size=(2,2)))#38*38*32

    model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))#36*36*64
    model.add(BatchNormalization())
    model.add(Dropout(rate=.2))

    model.add(MaxPooling2D(pool_size=(2,2)))#18*18*64

    model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu'))#16*16*64
    model.add(BatchNormalization())
    model.add(Dropout(rate=.2))

    model.add(Flatten())

    model.add(Dense(units=100,activation='relu',kernel_initializer='HeNormal'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=.2))
    model.add(Dense(units=config['no_students'],activation='softmax'))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    return model
