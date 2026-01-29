#!/usr/bin/env python3
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
from demo_programs.msg import prox_sensor

# Position: Backwards is 0, Left is -pi/2, Right is pi/2.
# Angle to move forward is -pi/2 to -pi AND pi/2 to pi

# "/cmd_vel": Velocity
# "/cop/pose": Position Data
class robotController:
    def __init__(self):
        velocityTopicName = '/cmd_vel'
        lidarTopicName = '/hokuyo'
        positionTopicName = '/cop/pose'
        proximityTopicName = 'cop/prox_sensors'
        
        self.velocityPublisher = rospy.Publisher(velocityTopicName, Twist, queue_size=10)
        
        self.velocitySubscriber = rospy.Subscriber(velocityTopicName, Twist, self.velocityCallback)
        self.lidarSubscriber = rospy.Subscriber(lidarTopicName, LaserScan, self.lidarCallback)
        self.positionSubscriber = rospy.Subscriber(positionTopicName, Pose, self.poseCallback)
        
        self.robotVelocity = Twist() 
        
        self.robotAngle = [0,0,0] # x,y,z
        self.robotPosition = [0,0,0] # x,y,z
        
        self.globalLeftAngle = -np.pi/2
        self.globalRightAngle = np.pi/2
        self.globalBackAngle = 0
        
    # Return 1 FORWARD RIGHT, return 2 BACK RIGHT, return -1 FORWARD LEFT, return -2 BACK LEFT,
    def __checkOrientation(self):
        if (self.robotAngle[2] > (np.pi/2)):
            return 1
        elif (self.robotAngle[2] < -(np.pi/2)):
            return -1
        elif (self.robotAngle[2] < 0):
            return -2
        return 2
    
    # LIDAR Index goes right to left
    def lidarCallback(self, msg):
        angleMin = msg.angle_min
        angleMax = msg.angle_max
        rangeMin = msg.range_min
        rangeMax = msg.range_max
        
        # Angle increment is 0.002*pi or 0.006 radians
        angleIncrement = msg.angle_increment
        
        lidarRanges = np.array(msg.ranges)
        
        # Array Lenth = 684 Length/2 = 342
        
        middleIndex = len(lidarRanges) // 2
        
        maxSensorRange = 5  # Replace with your LIDAR's maximum range
        lidarRanges[np.isinf(lidarRanges)] = maxSensorRange
        lidarRanges[np.isnan(lidarRanges)] = 0.0
        
        # Validate Ranges
        # Robot facing left side
        validIndexLeft = (self.globalLeftAngle - self.robotAngle[2]) // angleIncrement
        validIndexRight = -(2*np.pi - abs(self.globalRightAngle - self.robotAngle[2])) // angleIncrement
    
        # Robot facing right side
        if (self.__checkOrientation() >= 0): 
            validIndexLeft = (2*np.pi - abs(self.globalLeftAngle - self.robotAngle[2])) // angleIncrement
            validIndexRight = (self.globalRightAngle - self.robotAngle[2]) // angleIncrement
        
        validLeft = int(middleIndex + validIndexLeft)
        if (validLeft > len(lidarRanges)): validLeft = len(lidarRanges)
        
        validRight = int(middleIndex + validIndexRight)
        if (validRight < 0): validRight = 0        
        
        validRanges = np.copy(lidarRanges)
        validRanges[validLeft:] = 0
        validRanges[:validRight] = 0
        
        # Find the direction with the maximum distance
        maxDistance = np.max(validRanges)
        maxIndex = np.argmax(validRanges)

        # Calculate the angle of the maximum distance
        maxAngle = angleMin + maxIndex * angleIncrement
        
        #rospy.loginfo(f"Angle Increment: {angleIncrement} Min Angle: {angleMin:.2f}")
        
        self.robotVelocity.linear.x = 0.5 # Keep at 0.5 
        self.robotVelocity.angular.z = -maxAngle * 1.5 
        
        #rospy.loginfo(f"Target Distance: {maxDistance:.3f} Target Index: {maxIndex}")
        
        # Publish the command
        self.velocityPublisher.publish(self.robotVelocity)
    
        return
    
    def poseCallback(self, msg):
        # Extract orientation (quaternion)
        orientation_q = msg.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    
        # Convert quaternion to Euler angles
        self.robotAngle[0], self.robotAngle[1], self.robotAngle[2] = euler_from_quaternion(orientation_list)
        self.robotPosition = [msg.position.x, msg.position.y, msg.position.z]
        
        #rospy.loginfo(f"x: {self.robotPosition[0]:.3f} y: {self.robotPosition[1]:.3f} Angle: {self.robotAngle[2]:.3f}")
        
        return
    
    def velocityCallback(self, msg):
        # Extract linear and angular velocity
        linear_x = msg.linear.x
        linear_y = msg.linear.y
        linear_z = msg.linear.z
    
        angular_x = msg.angular.x
        angular_y = msg.angular.y
        angular_z = msg.angular.z
    
        # Log the velocities
        rospy.loginfo(f"Linear Velocity -> x: {linear_x}, y: {linear_y}, z: {linear_z}")
        rospy.loginfo(f"Angular Velocity -> x: {angular_x}, y: {angular_y}, z: {angular_z:.3f}")
        return
    
    def velocityUpdater(self):
        return
    
def main():
    nodeName = "robotController"
    rospy.init_node(nodeName)
    robot = robotController()
    
    refreshRate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        robot.velocityUpdater()
        
        refreshRate.sleep()
    #rospy.spin()
    
if __name__ == '__main__':
    main()
    