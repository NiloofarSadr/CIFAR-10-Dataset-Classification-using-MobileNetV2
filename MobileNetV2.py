"""
Zeinab Sadat (Niloofar) Sadrolhefazi
e_mail: zns.sadr@gmail.com
"""

import tensorflow as tf
import time
from  matplotlib import pyplot as plt
from tensorflow.keras import  layers
from tensorflow.keras import  models
from tensorflow.keras import  optimizers
from tensorflow.keras import  datasets
from tensorflow.keras.applications import MobileNetV2
pre_trained_model = MobileNetV2(weights='imagenet',
                                include_top=False,
                                input_shape=(32, 32, 3))
model = models.Sequential()
model.add(pre_trained_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

pre_trained_model.trainable = False



(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

epoch_num = 2

start_time = time.time()


model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy',metrics=['accuracy'])
hist = model.fit(x_train,y_train,epochs =epoch_num, validation_split=0.2, shuffle=True)
#hist = cnn.fit(x_train,y_train,epochs =epoch_num, validation_split=0.2, shuffle=True)


print("--- %s seconds ---" % str((time.time() - start_time)/epoch_num))
print('Test',model.evaluate(x_test,y_test))



plt.plot(hist.history['loss'], linestyle = 'dotted',label = 'Train')
plt.plot(hist.history['val_loss'], linestyle = 'dotted',label = 'Validation')
plt.title('Loss')
plt.legend()
plt.show()



