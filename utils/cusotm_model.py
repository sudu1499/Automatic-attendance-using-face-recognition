from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Flatten
from keras.models import Sequential

def custom_model(config):

    model=Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='valid',input_shape=(160,160,3)))#158
    model.add(Dropout(rate=.2))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))#79
    
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='valid'))#77
    model.add(Dropout(rate=.2))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))#38
    
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='valid'))#37
    model.add(Dropout(rate=.2))
    model.add(BatchNormalization())
   
    model.add(MaxPooling2D(pool_size=(2,2)))#18
    
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='valid'))#16
    model.add(Dropout(rate=.2))
    model.add(BatchNormalization())

    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='valid'))#14

    model.add(Flatten())
    model.add(Dense(50,activation='relu',kernel_initializer='HeNormal'))
    model.add(Dropout(rate=.2))
    model.add(BatchNormalization())
    
    model.add(Dense(50,activation='relu',kernel_initializer='HeNormal'))
    model.add(Dropout(rate=.2))
    model.add(BatchNormalization())

    model.add(Dense(config['no_students'],activation='softmax',kernel_initializer='HeNormal'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

    return model
    

    