#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Load required packages
import os
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import rospy
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt

# Avoid potential OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# List of arrow classes
namesList = ['up', 'down', 'left', 'right']

class ArrowRobotController:
    def __init__(self, modelPath="arrow_cnn_model4.h5", validationPath="validation"):
        """Initialize the controller, load the CNN model, and set up the ROS node."""
        self.modelPath = os.path.join(os.path.dirname(__file__), modelPath)
        self.validationPath = os.path.join(os.path.dirname(__file__), validationPath)
        self.namesList = ['up', 'down', 'left', 'right']
        
        rospy.init_node('arrow_control_node', anonymous=True)
        self.velocityPublisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        self.model = self.loadModel()
        
        # Set initial Image Sizes
        self.updateImageSize = [128, 128]
        [self.imWidth, self.imHeight] = 0,0
        self.firstImage = False

        self.predictedImage = 0
        self.image = 0
        self.trueDirection = 0
    
    def loadRandomArrow(self):
        """Select a random arrow image from one of the 'up', 'down', 'left', or 'right' folders."""
        direction = random.choice(self.namesList)  # Randomly select one of the four directions
        imagesList = glob.glob(os.path.join(self.validationPath, direction, "*.jpg"))
        imagePath = random.choice(imagesList)  # Get a random image from the selected folder
        self.predictedImage = Image.open(imagesList[0]).convert('L')
        
        image = Image.open(imagePath).convert('L')  # Convert to grayscale
        self.image = image
        
        image.thumbnail(self.updateImageSize, Image.ANTIALIAS)
        if not self.firstImage:
            [self.imWidth, self.imHeight] = image.size
            self.firstImage = True
        imageArray = np.array(image, 'f') / 255.0  # Normalize the image

        # Reshape the image to the shape expected by the CNN model
        imageArray = np.reshape(imageArray, (1, self.imHeight, self.imWidth, 1))  # Add batch dimension
        return imageArray, direction
    
    def loadModel(self):
        """Load the pre-trained Keras CNN model."""
        print("Loading CNN model...")
        model = load_model(self.modelPath)
        print("Model loaded successfully.")
        return model
    
    def sendMovement(self, direction):
        """Send a velocity command to the robot based on the arrow direction."""
        twist = Twist()
        
        if direction == 'up':
            twist.linear.x = 0.5  # Move forward#
        elif direction == 'down':
            twist.linear.x = -0.5  # Move backward
        elif direction == 'left':
            twist.angular.z = -0.5  # Turn left
            twist.linear.x = 0.5
        elif direction == 'right':
            twist.angular.z = 0.5  # Turn right
            twist.linear.x = 0.5
        
        # Plot the predicted image with its label
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.imshow(self.predictedImage, cmap='gray')
        plt.title(f"Predicted: {direction}")
        plt.axis('off')  # Hide axes
        
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        plt.imshow(self.image, cmap='gray')  # Same image, but showing the actual label
        plt.title(f"Actual: {self.trueDirection}")
        plt.axis('off')  # Hide axes
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)
        
        
        print(f"Moving {direction} for 2 seconds...")
        rate = rospy.Rate(10)  # 10 Hz loop rate
        for i in range(20):  # Send command for 2 seconds (10 Hz * 2 seconds = 20 loops)
            self.velocityPublisher.publish(twist)
            rate.sleep()
        plt.close()
    
    def run(self):
        """Main program loop to detect arrow direction and send movement commands to the robot."""
        try:
            while not rospy.is_shutdown():
                # Load a random arrow image
                imageArray, self.trueDirection = self.loadRandomArrow()
                
                # Use the CNN model to predict the direction
                prediction = self.model.predict(imageArray)
                predictedIndex = np.argmax(prediction)  # Get the index of the highest probability
                predictedDirection = self.namesList[predictedIndex]
                
                print(f"True direction: {self.trueDirection}, Predicted direction: {predictedDirection}")
                
                # Send movement command to the robot
                self.sendMovement(predictedDirection)
        
        except rospy.ROSInterruptException:
            print("ROS node interrupted. Shutting down.")


if __name__ == "__main__":
    controller = ArrowRobotController()
    controller.run()
