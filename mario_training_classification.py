import keras
import time
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import RMSprop, SGD, Adam
from keras.models import load_model
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import transform
import sys

import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
import actions
from actions import COMPLEX_MOVEMENT
train = os.getcwd() + "\\train\\"
# test = os.getcwd() + "\\test\\"
train_directories = [train+i for i in os.listdir(train) if 'capture' in i]
# test_directories = [test+i for i in os.listdir(test) if 'capture' in i]

csv_train = []
train_images = []

# csv_test = []
# test_images = []

for _train in train_directories:
    csv_train.append(pd.read_csv(_train + "\\controller_capture.csv"))
    train_images += [_train + "\\" + i for i in os.listdir(_train) if '.png' in i]
csv_train = pd.concat(csv_train, axis=0, ignore_index=True)

# for _test in test_directories:
#     csv_test.append(pd.read_csv(_test + "\\controller_capture.csv"))
#     test_images += [_test + "\\" + i for i in os.listdir(_test) if '.png' in i]
# csv_test = pd.concat(csv_test, axis=0, ignore_index=True)

training = True

ROWS = 100
COLS = 120
CHANNELS = 1
input_shape = (ROWS, COLS, CHANNELS)
def convert_to_gray_scale(img):
    return np.dot(img, [0.299, 0.587, 0.114])


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    img2 = cv2.resize(img2, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    img2 = np.reshape(img2, (ROWS, COLS, CHANNELS))
    return img2

def read_image_gray(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE
    # b, g, r = cv2.split(img)
    # img2 = cv2.merge([r, g, b])
    # img2 = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    # img2 = np.reshape(img2, (ROWS, COLS, CHANNELS))
    # img2 = rgb2gray(img)
    # img2 = convert_to_gray_scale(img2)
    # plt.imshow(img2, cmap = 'gray')
    # plt.show()
    # img2 = img / 255.0 # Normalising the image
    # normalizedImg = np.zeros((800, 800))
    # img2 = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    
    img2 = cv2.resize(img, (COLS, ROWS + 28), interpolation=cv2.INTER_CUBIC)
    img2 = img2[28:128, 0:120]
    # print(img2.shape)
    # plt.imshow(img2, cmap = 'gray')
    # plt.show()
    img2 = img2 / 255.0
    # img2 = transform.resize(img2, [ROWS, COLS])
    return img2


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i % 250 == 0: print('Processed {} of {}'.format(i, count))

    return data

def prep_data_gray(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, 1), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image_gray(image_file)
        image = np.expand_dims(image, axis=2)
        data[i] = image
        if i % 250 == 0: print('Processed {} of {}'.format(i, count))

    return data

def create_model():
    optimizer = Adam(lr=0.00025)
    objective = 'binary_crossentropy'

    model = Sequential()

    model.add(Conv2D(32, (3, 3), border_mode='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), border_mode='same', activation='relu'))

    model.add(Conv2D(64, (3, 3), border_mode='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), border_mode='same', activation='relu'))
    # model.add(Conv2D(256, (3, 3), border_mode='same', activation='relu')) # Added line in
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model
    
def create_model_mnist_fashion():
    optimizer = Adam(lr=0.00025)
    objective = 'binary_crossentropy'
    
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss=objective,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

if training:
    model = create_model()
    # train = prep_data(train_images)
    # test = prep_data(test_images)

    train = prep_data_gray(train_images)
    # test = prep_data_gray(test_images)

    # input_shape = read_image(train_images[0]).shape
    # input_shape = (ROWS, COLS, CHANNELS)

    csv_train = csv_train.drop(csv_train.columns[0], axis=1)
    # csv_test = csv_test.drop(csv_test.columns[0], axis=1)

    model.fit(train, csv_train, epochs=400, verbose=1, shuffle=True)
    
    # print(model.evaluate(test, csv_test, verbose=1))

    model.save("model/mario.save")
else:
    model = load_model("model/mario.save")




environmentsmario = ["SuperMarioBros-v0",
                     "SuperMarioBros-v1",
                     "SuperMarioBros-v2",
                     "SuperMarioBros-v3",
                     "SuperMarioBrosNoFrameskip-v0",
                     "SuperMarioBrosNoFrameskip-v1",
                     "SuperMarioBrosNoFrameskip-v2",
                     "SuperMarioBrosNoFrameskip-v3",
                     "SuperMarioBros2-v0",
                     "SuperMarioBros2-v1",
                     "SuperMarioBros2NoFrameskip-v0",
                     "SuperMarioBros2NoFrameskip-v1"]

movements = [COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY]

image_shape = (ROWS, COLS, CHANNELS)


def createenvironment(enviro, movementset):
    environment = gym_super_mario_bros.make(enviro)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movementset)

    return environment


def startemulator(env, model, image_shape):
    fitness = []
    done = True
    env.reset()
    old_x_pos = sys.maxsize
    for step in range(5000):
        time.sleep(0.05)
        env.render()
        image = env.render('rgb_array')
        image = cv2.resize(image, dsize=(image_shape[1], image_shape[0] + 28), interpolation=cv2.INTER_CUBIC)
        image = convert_to_gray_scale(image)
        image = image[28:128, 0:120]
        # print(image.shape)
        # plt.imshow(image, cmap = 'gray')
        # plt.show()
        image = image / 255.0
        image = np.expand_dims(image, axis=2)
        image = np.expand_dims(image, axis=0)
        if done or step == 0:
            print("Fitness: " + str(np.sum(fitness)))
            fitness = []
            env.reset()
        action = model.predict(image)
        action = acions.calculate_action_list(action[0])
        action = acions.calculate_action_num(action)
        state, reward, done, info = env.step(action)
        if step % 120 == 0:
            if info['x_pos'] == old_x_pos:
                done = True
                old_x_pos = sys.maxsize
            else:
                old_x_pos = info['x_pos']
        fitness.append(reward)
        # print(info)

    env.close()


env = createenvironment(environmentsmario[0], movements[0])

startemulator(env, model, input_shape)
