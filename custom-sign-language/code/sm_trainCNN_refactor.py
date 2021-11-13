import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import itertools
import os
import json
import random
import warnings
import numpy as np
import cv2
import datetime

import matplotlib.pyplot as plt
#from utils.definitions import word_dict
import argparse
warnings.simplefilter(action='ignore', category=FutureWarning)

word_dict = {
 0:'One second, please.',
 1:'Hello. My name is Joel; I have an issue with my home internet network. It is extremely slow!',
 2:'Thank you. My user id is zero five one five one two three four five six seven eight.',
 3:'Should be good to go; testing it now.',
 4:'It works much better! Thanks a lot!',
 5:'Have a nice day. Bye!'
 }

#train_path = r'../images/train'
#test_path = r'../images/test'

train_path = ""
test_path = ""
train_batches = None
test_batches = None

def init_datasets(train_path, test_path):
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(64,64), class_mode='categorical', batch_size=10,shuffle=True)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(64,64), class_mode='categorical', batch_size=10, shuffle=True)
    #imgs, labels = next(train_batches)
    return train_batches, test_batches

#plotImages(imgs)
#print(imgs.shape)
#print(labels)

#Plotting the images...
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    #parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    return parser.parse_known_args()


def build_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())

    model.add(Dense(64,activation ="relu"))
    model.add(Dense(128,activation ="relu"))
    #model.add(Dropout(0.2))
    model.add(Dense(128,activation ="relu"))
    #model.add(Dropout(0.3))
    #model.add(Dense(10,activation ="softmax")) TODO return to 10
    model.add(Dense(6,activation ="softmax"))

    # In[23]:
    #imgs, labels = next(train_batches) #??

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


    model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    history2 = model.fit(train_batches, epochs=20, callbacks=[reduce_lr, early_stop,tensorboard],  validation_data = test_batches)#, checkpoint])
    #history2 = model.fit(train_batches, epochs=6, callbacks=[reduce_lr,tensorboard],  validation_data = test_batches)#, checkpoint])
    imgs, labels = next(train_batches) # For getting next batch of imgs...

    imgs, labels = next(test_batches) # For getting next batch of imgs...
    scores = model.evaluate(imgs, labels, verbose=0)
    print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

    #model.save('../model/best_model_custom_gestures.h5')
    print(history2.history)
    model.summary()

    scores #[loss, accuracy] on test data...
    model.metrics_names
    #predict()

def predict():
    print ("#### Start prediction... ####")
    imgs, labels = next(test_batches)

    model = tf.keras.models.load_model(r"../model/best_model_custom_gestures.h5")

    scores = model.evaluate(imgs, labels, verbose=0)
    print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    predictions = model.predict(imgs, verbose=0)
    print("predictions on a small set of test data--")
    print("")
    for ind, i in enumerate(predictions):
        print(word_dict[np.argmax(i)], end='   ')

    plotImages(imgs)
    print('Actual labels')
    for i in labels:
        print(word_dict[np.argmax(i)], end='   ')

    #history2.history

if __name__ == "__main__":
    #print ("*****************current dir= " + os.path.dirname(os.path.realpath(__file__)))

    print("*****************current dir= " + os.getcwd())
    args, unknown = _parse_args()
    train_path = args.train
    test_path = args.test

    print("train_path===>" + train_path)
    print("test_path===>" + test_path)
    train_batches, test_batches = init_datasets(train_path, test_path)
    build_model()