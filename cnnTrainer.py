# -*- coding: utf-8 -*-
"""
Detection of arrows for robot control using CNN and ROS

@author: Uriel Martinez-Hernandez
"""

# Load required packages
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from PIL import Image                                                            
import glob
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

np.random.seed(7)

# List of arrow classes
namesList = ['up', 'down', 'left', 'right']

# Folder names of train and testing images
imageFolderTrainingPath = './train'
imageFolderTestingPath = './validation'
imageTrainingPath = []
imageTestingPath = []

# Print number of images for training and testing
print("imageFolderTrainingPath: ", imageFolderTrainingPath)
print("imageFolderTestingPath: ", imageFolderTestingPath)

for i in range(len(namesList)):
    # Using os.path.join for better handling of paths
    trainingLoad = os.path.join(imageFolderTrainingPath, namesList[i], '*.jpg')
    testingLoad = os.path.join(imageFolderTestingPath, namesList[i], '*.jpg')

    # Print glob pattern for debugging
    print(f"Looking for training images in: {trainingLoad}")
    print(f"Looking for testing images in: {testingLoad}")

    imageTrainingPath = imageTrainingPath + glob.glob(trainingLoad)
    imageTestingPath = imageTestingPath + glob.glob(testingLoad)

# Debug print
print(f"Number of training images: {len(imageTrainingPath)}")
print(f"Number of testing images: {len(imageTestingPath)}")

# If the list is empty, check if glob found anything
if len(imageTrainingPath) == 0:
    print("No training images found.")
if len(imageTestingPath) == 0:
    print("No testing images found.")

# Resize images to speed up the training process
updateImageSize = [128, 128]
tempImg = Image.open(imageTrainingPath[0]).convert('L')
tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
[imWidth, imHeight] = tempImg.size

# Create space to load training and testing images
x_train = np.zeros((len(imageTrainingPath), imHeight, imWidth, 1))
x_test = np.zeros((len(imageTestingPath), imHeight, imWidth, 1))

# Load training images
for i in range(len(x_train)):
    tempImg = Image.open(imageTrainingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_train[i, :, :, 0] = np.array(tempImg, 'f')

# Load testing images
for i in range(len(x_test)):
    tempImg = Image.open(imageTestingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_test[i, :, :, 0] = np.array(tempImg, 'f')

# Normalize image pixel values
x_train /= 255.0
x_test /= 255.0

# Create space for training and testing labels
y_train = np.zeros((len(x_train),))
y_test = np.zeros((len(x_test),))

# Load training labels
countPos = 0
for i in range(len(namesList)):
    for j in range(round(len(imageTrainingPath)/len(namesList))):
        y_train[countPos,] = i
        countPos += 1

# Load testing labels
countPos = 0
for i in range(len(namesList)):
    for j in range(round(len(imageTestingPath)/len(namesList))):
        y_test[countPos,] = i
        countPos += 1

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, len(namesList))
y_test = tf.keras.utils.to_categorical(y_test, len(namesList))

        

# Creat your CNN model here composed of convolution, maxpooling, fully connected layers.

# Compile, fit and evaluate your CNN model.

# Stores your model in a specific path

# Display the accuracy achieved by your CNN model

# Plot accuracy plots

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(imHeight, imWidth, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    #Dense(len(namesList))
    Dense(len(namesList), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x_train, y_train, 
                    epochs=20, 
                    batch_size=32, 
                    validation_data=(x_test, y_test))

model.save('arrow_cnn_model4.h5')
model.save('arrow_cnn_model4.keras')

plt.figure(figsize=(10, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the final model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

print('OK')