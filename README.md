# Jump Height Tracker

## Project Description
The Jump Height Tracker is an application developed in Python that uses computer vision techniques to track and measure the height of a person's jumps in real-time. It utilizes the MediaPipe library for pose detection and OpenCV for image processing and display.

## How Jump Height is Calculated
The jump height is calculated using the following steps:
1. **Calibration**: The user stands in front of the camera in a T-pose. The application calibrates the standing position by measuring the y-coordinates of the hips in this pose.
2. **Detecting the Jump**: The application continuously tracks the y-coordinates of the user's hips. A jump is detected when there is a noticeable upward movement from the calibrated standing position.
3. **Measuring Jump Height**: During the jump, the application measures the difference between the y-coordinate of the hips at the peak of the jump and their position in the calibrated standing pose. This difference, converted from pixels to centimeters, represents the jump height.
4. **Recording Jump Heights**: Each jump height is recorded and written to a text file for later reference.
5. **Resetting After Landing**: Once the user lands and the hips return to a position close to the calibrated standing position, the application resets, ready to measure the next jump.

## Requirements
To run the Jump Height Tracker, you need to have the following installed:
- Python 3.6 or higher
- OpenCV library
- MediaPipe library

You can install the required libraries using the following commands:

pip install opencv-python
pip install mediapipe


### Running the Application
To start the application, navigate to the project directory in your command line and run:
python jump.py
