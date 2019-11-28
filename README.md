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



Logs:
Epoch 1/50
390/390 [==============================] - 35s 89ms/step - loss: 1.5428 - acc: 0.4386 - val_loss: 1.3881 - val_acc: 0.5471
Epoch 2/50
390/390 [==============================] - 32s 81ms/step - loss: 1.0776 - acc: 0.6194 - val_loss: 1.0637 - val_acc: 0.6349
Epoch 3/50
390/390 [==============================] - 32s 82ms/step - loss: 0.9414 - acc: 0.6687 - val_loss: 0.9398 - val_acc: 0.6815
Epoch 4/50
390/390 [==============================] - 32s 81ms/step - loss: 0.8659 - acc: 0.6936 - val_loss: 1.2080 - val_acc: 0.6107
Epoch 5/50
390/390 [==============================] - 32s 81ms/step - loss: 0.8074 - acc: 0.7168 - val_loss: 0.8646 - val_acc: 0.6987
Epoch 6/50
390/390 [==============================] - 32s 81ms/step - loss: 0.7642 - acc: 0.7323 - val_loss: 0.8704 - val_acc: 0.7016
Epoch 7/50
390/390 [==============================] - 32s 81ms/step - loss: 0.7331 - acc: 0.7423 - val_loss: 0.7639 - val_acc: 0.7347
Epoch 8/50
390/390 [==============================] - 32s 82ms/step - loss: 0.6952 - acc: 0.7566 - val_loss: 0.9276 - val_acc: 0.6855
Epoch 9/50
390/390 [==============================] - 31s 81ms/step - loss: 0.6758 - acc: 0.7626 - val_loss: 0.8072 - val_acc: 0.7261
Epoch 10/50
390/390 [==============================] - 31s 81ms/step - loss: 0.6573 - acc: 0.7716 - val_loss: 0.9680 - val_acc: 0.6758
Epoch 11/50
390/390 [==============================] - 31s 81ms/step - loss: 0.6343 - acc: 0.7776 - val_loss: 0.7249 - val_acc: 0.7528
Epoch 12/50
390/390 [==============================] - 31s 81ms/step - loss: 0.6149 - acc: 0.7865 - val_loss: 0.7314 - val_acc: 0.7498
Epoch 13/50
390/390 [==============================] - 31s 81ms/step - loss: 0.6039 - acc: 0.7896 - val_loss: 0.7609 - val_acc: 0.7415
Epoch 14/50
390/390 [==============================] - 31s 80ms/step - loss: 0.5901 - acc: 0.7937 - val_loss: 0.6696 - val_acc: 0.7740
Epoch 15/50
390/390 [==============================] - 31s 80ms/step - loss: 0.5724 - acc: 0.7990 - val_loss: 0.9164 - val_acc: 0.6926
Epoch 16/50
390/390 [==============================] - 32s 81ms/step - loss: 0.5652 - acc: 0.8027 - val_loss: 0.6624 - val_acc: 0.7719
Epoch 17/50
390/390 [==============================] - 31s 80ms/step - loss: 0.5429 - acc: 0.8101 - val_loss: 0.6991 - val_acc: 0.7608
Epoch 18/50
390/390 [==============================] - 32s 81ms/step - loss: 0.5352 - acc: 0.8131 - val_loss: 0.6232 - val_acc: 0.7888
Epoch 19/50
390/390 [==============================] - 31s 80ms/step - loss: 0.5340 - acc: 0.8124 - val_loss: 0.6946 - val_acc: 0.7678
Epoch 20/50
390/390 [==============================] - 32s 81ms/step - loss: 0.5223 - acc: 0.8176 - val_loss: 0.6235 - val_acc: 0.7869
Epoch 21/50
390/390 [==============================] - 31s 81ms/step - loss: 0.5099 - acc: 0.8222 - val_loss: 0.6029 - val_acc: 0.7927
Epoch 22/50
390/390 [==============================] - 31s 80ms/step - loss: 0.5019 - acc: 0.8257 - val_loss: 0.6386 - val_acc: 0.7777
Epoch 23/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4909 - acc: 0.8272 - val_loss: 0.8192 - val_acc: 0.7229
Epoch 24/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4888 - acc: 0.8291 - val_loss: 0.6552 - val_acc: 0.7674
Epoch 25/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4813 - acc: 0.8307 - val_loss: 0.6089 - val_acc: 0.7898
Epoch 26/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4745 - acc: 0.8339 - val_loss: 0.6612 - val_acc: 0.7734
Epoch 27/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4644 - acc: 0.8364 - val_loss: 0.6419 - val_acc: 0.7794
Epoch 28/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4563 - acc: 0.8402 - val_loss: 0.7225 - val_acc: 0.7569
Epoch 29/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4569 - acc: 0.8402 - val_loss: 0.5772 - val_acc: 0.8058
Epoch 30/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4470 - acc: 0.8415 - val_loss: 0.6514 - val_acc: 0.7766
Epoch 31/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4459 - acc: 0.8427 - val_loss: 0.6366 - val_acc: 0.7853
Epoch 32/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4355 - acc: 0.8465 - val_loss: 0.5780 - val_acc: 0.8041
Epoch 33/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4342 - acc: 0.8461 - val_loss: 0.5467 - val_acc: 0.8099
Epoch 34/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4252 - acc: 0.8508 - val_loss: 0.6242 - val_acc: 0.7924
Epoch 35/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4207 - acc: 0.8509 - val_loss: 0.6594 - val_acc: 0.7749
Epoch 36/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4181 - acc: 0.8541 - val_loss: 0.5863 - val_acc: 0.8007
Epoch 37/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4135 - acc: 0.8519 - val_loss: 0.7215 - val_acc: 0.7606
Epoch 38/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4144 - acc: 0.8542 - val_loss: 0.7143 - val_acc: 0.7641
Epoch 39/50
390/390 [==============================] - 31s 80ms/step - loss: 0.4049 - acc: 0.8567 - val_loss: 0.7425 - val_acc: 0.7503
Epoch 40/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3962 - acc: 0.8604 - val_loss: 0.6112 - val_acc: 0.7936
Epoch 41/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3953 - acc: 0.8610 - val_loss: 0.5783 - val_acc: 0.7993
Epoch 42/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3954 - acc: 0.8606 - val_loss: 0.5961 - val_acc: 0.7924
Epoch 43/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3840 - acc: 0.8639 - val_loss: 0.5724 - val_acc: 0.8071
Epoch 44/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3917 - acc: 0.8611 - val_loss: 0.5696 - val_acc: 0.8078
Epoch 45/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3827 - acc: 0.8653 - val_loss: 0.5885 - val_acc: 0.7994
Epoch 46/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3811 - acc: 0.8661 - val_loss: 0.6105 - val_acc: 0.7900
Epoch 47/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3764 - acc: 0.8666 - val_loss: 0.5865 - val_acc: 0.8054
Epoch 48/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3717 - acc: 0.8696 - val_loss: 0.6159 - val_acc: 0.7944
Epoch 49/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3752 - acc: 0.8666 - val_loss: 0.5494 - val_acc: 0.8170
Epoch 50/50
390/390 [==============================] - 31s 80ms/step - loss: 0.3637 - acc: 0.8697 - val_loss: 0.5849 - val_acc: 0.8020
Model took 1572.65 seconds to train

Accuracy on test data is: 80.20
