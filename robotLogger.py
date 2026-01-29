#!/usr/bin/env python3
import rospy
import csv
import os
from datetime import datetime
from std_msgs.msg import Bool  # Import Bool message type
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from demo_programs.msg import prox_sensor, line_sensor

class RobotLogger:
    def __init__(self):
        self.logFilePath = os.path.join(os.path.expanduser("~"), "robot_log.csv")
        self.__initializeLogFile()

        # Data storage variables
        self.robotVelocity = Twist()
        self.robotLinearPosition = [0, 0, 0]
        self.robotAngularPosition = [0, 0, 0]
        self.lidarData = []
        self.proximityData = {'front_left_left': 0, 'front_right_right': 0}
        self.lineSensorData = {'line_left': 0, 'line_middle': 0, 'line_right': 0}
        self.robotStop = False 

        # Store start time for elapsed time calculation
        self.startTime = rospy.get_rostime()
        self.stopTimeLogged = False  # Flag to log the first instance of robotStop being True

        # Subscribers
        rospy.Subscriber('/cmd_vel', Twist, self.velocityCallback)
        rospy.Subscriber('/cop/pose', Pose, self.poseCallback)
        rospy.Subscriber('/hokuyo', LaserScan, self.lidarCallback)
        rospy.Subscriber('cop/prox_sensors', prox_sensor, self.proximityCallback)
        rospy.Subscriber('cop/line_sensors', line_sensor, self.lineCallback)
        rospy.Subscriber('/robot_stop_status', Bool, self.robotStopCallback)  # NEW: Subscribe to robot stop status
        
        rospy.Timer(rospy.Duration(0.5), self.__logData)

    def __initializeLogFile(self):
        """Creates or clears the log file and writes the header."""
        with open(self.logFilePath, 'w', newline='') as file:
            writer = csv.writer(file)
            header = [
                'Real_Timestamp', 'Sim_Timestamp', 'Pos_X', 'Pos_Y', 'Angle_Yaw',
                'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Z',
                'Prox_Front_Left_Left', 'Prox_Front_Right_Right',
                'Robot_Stop'  # Added Robot_Stop
            ]
            writer.writerow(header)
        rospy.loginfo(f"Log file initialized at {self.logFilePath}")

    def __logData(self, event):
        """Logs the current state of the robot to a CSV file."""
        with open(self.logFilePath, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Calculate elapsed time from node start
            currentTime = rospy.get_rostime()
            elapsedTime = (currentTime - self.startTime).to_sec()
            
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),  # Real-world timestamp
                round(elapsedTime, 3),  # Elapsed time from start (3 decimal places)
                round(self.robotLinearPosition[0], 3),  # Position X
                round(self.robotLinearPosition[1], 3),  # Position Y
                round(self.robotAngularPosition[2], 3),  # Yaw angle
                round(self.robotVelocity.linear.x, 3),  # Linear velocity X
                round(self.robotVelocity.linear.y, 3),  # Linear velocity Y
                round(self.robotVelocity.angular.z, 3),  # Angular velocity Z
                round(self.proximityData['front_left_left'], 3),  # Proximity front left left
                round(self.proximityData['front_right_right'], 3),  # Proximity front right right
                self.robotStop  # Robot stop status (True/False)
            ]
            writer.writerow(row)
            
            # Log the time when robotStop becomes True (only the first time)
            if self.robotStop and not self.stopTimeLogged:
                rospy.loginfo(f"Robot stopped. Elapsed time: {elapsedTime:.3f} seconds")
                with open(self.logFilePath, 'a', newline='') as stop_file:
                    stopWriter = csv.writer(stop_file)
                    stopWriter.writerow(['Robot Stopped', f'{elapsedTime:.3f} seconds'])
                self.stopTimeLogged = True  # Ensure we only log it once

    def robotStopCallback(self, msg):
        """Updates the latest robot stop status."""
        self.robotStop = msg.data

    def velocityCallback(self, msg):
        self.robotVelocity = msg

    def poseCallback(self, msg):
        self.robotLinearPosition = [msg.position.x, msg.position.y, msg.position.z]
        from tf.transformations import euler_from_quaternion
        orientation_list = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.robotAngularPosition = euler_from_quaternion(orientation_list)

    def lidarCallback(self, msg):
        lidarRanges = list(msg.ranges)
        self.lidarData = lidarRanges

    def proximityCallback(self, msg):
        self.proximityData = {'front_left_left': msg.prox_front_left_left, 'front_right_right': msg.prox_front_right_right}

    def lineCallback(self, msg):
        self.lineSensorData = {'line_left': msg.line_left, 'line_middle': msg.line_middle, 'line_right': msg.line_right}


def main():
    rospy.init_node('robot_logger', anonymous=True)
    RobotLogger()
    rospy.spin()


if __name__ == '__main__':
    main()
