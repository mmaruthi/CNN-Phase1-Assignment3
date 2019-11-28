# CNN-Phase1-Assignment3
phase1-Assignment3:

    A)  Final Validation on Base Network is 81.34 
    Epoch 49/50
390/390 [==============================] - 7s 19ms/step - loss: 0.3564 - acc: 0.8810 - val_loss: 0.5684 - val_acc: 0.8245
Epoch 50/50
390/390 [==============================] - 7s 18ms/step - loss: 0.3559 - acc: 0.8801 - val_loss: 0.6114 - val_acc: 0.8134
Model took 365.30 seconds to train

Accuracy on test data is: 81.34

   B) Approach for new code :
      > Used Depth wise Seperable convolution - Separableconvolution2D
      > Used Dilation but i do not see it as an accuracy booster 
      > Used Bath Normalization 
      > used a drop out of 0.25 
      > Used Max pooling 
      > Used a batch size of 32 and learnin rate of 0.003 
      > PARAMS came up to 96k 
      
      Best test Accuracy received is 
