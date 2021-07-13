import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2 # image operations
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

'''SORTING DATA WHEN ITS NOT ALREADY IN KERAS
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(data_dir, category) # goes into specified category in given path
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path): # lists all image in current location
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # Converts inmage to array, must specify grAYSCALE!
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                training_data.append([new_array,class_num])
            except Exception as e:
                pass # Pass the images that may be problematic

# Setup -----------

data_dir = "/Users/queenro/PycharmProjects/Digit_Predictor/shapes"

CATEGORIES = ["circles", "triangles"]

training_data = []

IMG_SIZE = 28
# ----------------------
create_training_data()

random.shuffle(training_data)

X = []  # will contain each input/image as a matrix/array
y = []
for features,label in training_data:  # Don't do this directly in create training data method in order to shuffle first
    X.append(features)
    y.append(label)


# ---------- Seperate into training and test
x_train, x_test = X[:70], X[70:]
y_train, y_test = y[:70], y[70:]

'''
mnist = tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()



# ------------- Visualizations
#print(X.shape())
#print (X[0])
#plt.imshow(X[0]) #makes graph
#plt.show() #executes graph

#plt.imshow(X[1], cmap=plt.cm.binary_r)
#plt.show()


# ----------- Normalize Data

x_train = tf.keras.utils.normalize(x_train, axis=1)  # Equivalent to x_train/255
x_test = tf.keras.utils.normalize(x_test,axis=1)
#print (x_train[0])
#print(y[0])

# ------------ Must reshape data before being fed to model. Model does not take lists!!!!!!
IMG_SIZE = 28
x_train = np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1) # -1 means "anything". Represents how many of feature sets/arrays we have
x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Increasing by one dimmension at end, necessary for kernel operation

# ------------- Creating Deep Neural Network

model = Sequential()

# First convolutional layer
model.add(Conv2D(64, (3, 3), input_shape=x_train.shape[1:]))  #  64 3x3 filters. Each filter will be "convolved"/multiplied with an image.
                                                              # The input_shapes skips the first coord. ie (300,28,28,1) -> (28,28,1)
model.add(Activation("relu"))  # activation function to make it linear

model.add(MaxPooling2D(pool_size=(2, 2))) # MaxPooling single maximum value of 2x2


# Second convolutional Layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional Layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layer
model.add(Flatten()) # i.e 20x20 becomes 400
model.add(Dense(64))
model.add(Activation("relu"))

# 2nd Fully Connected Layer
model.add(Dense(32))
model.add(Activation("relu"))

# 3rd Fully Connected Layer
model.add(Dense(10))
model.add(Activation("softmax"))


# ---------------------- Compile & Train
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )

model.fit(x_train, y_train, epochs=5, validation_split=0.3)  # Train model


'''
print(X.shape)
np.save('features.npy', X) #saving X you did all this for

np.save('labels.npy', y)
'''

model.save('digit.model')
np.save('x_test.npy', x_test)