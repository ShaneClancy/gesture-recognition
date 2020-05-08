#PyQt imports
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit 
from PyQt5.QtCore import QThread

import sys
import os
from time import sleep

import pyrealsense2 as rs;

import numpy as np;

import cv2 as cv;

import pickle

import gzip

import threading

from time import sleep

class WorkerThread(QThread):
    def __init__(self, containerObject):
        QThread.__init__(self)
        self.containerObject = containerObject

    def run(self):
        while(True):
            if (self.containerObject.captureButton.text() ==  'Stop Capturing'):
                self.containerObject.captureImage()
                sleep(3)
            else:
                sleep(.5)

class Container(QWidget):

    def __init__(self):
        super().__init__();
        print('Loading model')
        self.model = self.loadModel()
        print('Model loaded')
        self.initUI();

    def initUI(self):
        # Capture button to initialize capturing of data.
        captureButton = QPushButton('Capture Data', self);
        captureButton.setToolTip('Use this button to capture an image from camera');
        captureButton.resize(150, 100);
        captureButton.move(300, 0);
        captureButton.clicked.connect(self.onCaptureClick);

    def onCaptureClick(self):
        print(3)
        sleep(1)
        print(2)
        sleep(1)
        print(1)
        sleep(1)
        print('Capturing')
        self.captureImage()

    def loadModel(self):
        return pickle.load(gzip.open('trained_model.pklz'))

    def removeBackground(self, image):
        if (np.max(image) != 0):

            minVal = np.min(image[np.nonzero(image)])

            image[image > 1500 + minVal] = 0

        return image

    def classifyImage(self, image):
        print(self.model.predict(image.copy().flatten().reshape(1, -1)))

    def captureImage(self):
        # Create a pipeline
        pipeline = rs.pipeline()

        #Create a config and configure the pipeline to stream for depth at 30fps
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start streaming pipeline
        profile = pipeline.start(config)

        # Getting depth sensor
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # Set a clipping distance of 1 meter
        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        # Align stream to depth
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Get frame of depth
        frames = pipeline.wait_for_frames();
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        # Compile a depth image
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        
        # Remove background
        depth_image = self.removeBackground(depth_image)
        
        # Turn image into three channel image
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image))

        # Create a colormap depth image
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        # Create gray image so we can find contours
        depth_gray = cv.cvtColor(depth_colormap, cv.COLOR_BGR2GRAY)
        canny_output = cv.Canny(depth_gray, 100, 200);
        contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Create copy to draw contours on
        contour_map = depth_colormap.copy()
        contour_map = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        # Draw contours
        for i in range(len(contours)):
            color = (0, 255, 0)
            cv.drawContours(contour_map, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        
        self.classifyImage(contour_map)

        # Draw images side by side
        images = np.hstack((depth_colormap, contour_map))
        cv.namedWindow('Contour Example', cv.WINDOW_AUTOSIZE)
        cv.imshow('Contour Example', images)
        key = cv.waitKey(1)



def main():
    app = QApplication(sys.argv);
    window = Container();
    window.setGeometry(0, 0, 450, 100);
    window.setWindowTitle("Gesture Recognition Training GUI");
    window.show();
    sys.exit(app.exec_());

if __name__ == "__main__":
    main();