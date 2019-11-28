# CNN-Phase1-Assignment3
phase1-Assignment3:

    A)  Final Validation on Base Network is 81.34 
    Epoch 49/50
390/390 [==============================] - 7s 19ms/step - loss: 0.3564 - acc: 0.8810 - val_loss: 0.5684 - val_acc: 0.8245
Epoch 50/50
390/390 [==============================] - 7s 18ms/step - loss: 0.3559 - acc: 0.8801 - val_loss: 0.6114 - val_acc: 0.8134
Model took 365.30 seconds to train

Accuracy on test data with baseline code is: 81.34

   B) Approach for new code :
      > Used Depth wise Seperable convolution - Separableconvolution2D
      > Used Dilation but i do not see it as an accuracy booster 
      > Used Bath Normalization 
      > used a drop out of 0.25 
      > Used Max pooling 
      > Used a batch size of 128 and learning rate of 0.003 
      > PARAMS came up to 96k 
      
      Best test Accuracy received is 81.70 which is better than the baseline version. 
      I got better accuracy with other codes touching 82.5.
      
      Model :
      from keras.layers import Activation

model = Sequential()
model.add(SeparableConv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))  #RF 3 , #Size 30
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(SeparableConv2D(64, kernel_size=(3, 3), activation='relu'))  #RF5  , #Size28
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(SeparableConv2D(128, kernel_size=(3, 3), activation='relu')) #RF7   , #Size26
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(SeparableConv2D(256, kernel_size=(3, 3), activation='relu')) #RF9   , #Size24
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))                               #RF10  , #Size12

model.add(SeparableConv2D(128, kernel_size=(3, 3), activation='relu'))  #RF14    , #size 10
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(SeparableConv2D(64, kernel_size=(3, 3), activation='relu'))   #RF18    , #Size 8
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(SeparableConv2D(32, kernel_size=(3, 3), activation='relu'))    #RF22   ,#Size6
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))                                #RF24   ,#Size3

model.add(SeparableConv2D(10, kernel_size=(3, 3), activation='relu'))    #RF32    ,#Size 1

model.add(Flatten())
model.add(Activation('softmax'))

model.summary()

# Compile the model

from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.003 * 1/(1 + 0.319 * epoch), 10)
model.compile(optimizer=Adam(lr=0.004), loss='categorical_crossentropy', metrics=['accuracy'])

