from __init__ import __version__
__author__ = "Ognjen Babovic, Lazar Gopcevic"
__copyright__ = "Ognjen Babovic, Lazar Gopcevic"
__license__ = "none"

import os, random
import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

### user defined modules
from iohelper import read_and_resize_image, show_loss_plot, show_image_plot
from constants import TRAIN_DIR, TEST_DIR, ROWS, COLS, CHANNELS, nb_epoch, batch_size


### classes

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


### functions

def prep_data(images, channels, rows, cols):
    count = len(images)
    data = np.ndarray((count, channels, rows, cols), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_and_resize_image(image_file, rows, cols)
        data[i] = image.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data

def catdog(rows, cols, objective, optimizer):
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, rows, cols), activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model

def run_catdog(model, train, labels, batch_size, nb_epoch, callback, test):
    history = LossHistory()
    model.fit(train, labels, batch_size=batch_size, epochs=nb_epoch,
              validation_split=0.25, verbose=1, shuffle=True, callbacks=[history, callback])
    
    predictions = model.predict(test, verbose=0)
    return predictions, history

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
train_images = train_dogs[:500] + train_cats[:500] # For testing using subset of data

random.shuffle(train_images)
test_images =  test_images[:50]

labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')        

train = prep_data(train_images, CHANNELS, ROWS, COLS)
test = prep_data(test_images, CHANNELS, ROWS, COLS)

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

model = catdog(ROWS, COLS, objective, optimizer)

predictions, history = run_catdog(model, train, labels, batch_size, nb_epoch, early_stopping, test)

loss = history.losses
val_loss = history.val_losses

show_loss_plot(loss, val_loss, nb_epoch)

for i in range(0,10):
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))
        
    show_image_plot(test[i].T)
