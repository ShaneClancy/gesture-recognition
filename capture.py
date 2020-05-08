# This module continually captures images from Intel RealSense SR305 Camera and finds the contours from the current image.

import pyrealsense2 as rs;

import numpy as np;

import cv2 as cv;

import random as rng;

def removeBackground(image):
    if (np.max(image) != 0):

        minVal = np.min(image[np.nonzero(image)])

        image[image > 1500 + minVal] = 0

    return image


def main():
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

    while(True):
        
        # Get frame of depth
        frames = pipeline.wait_for_frames();
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        # Compile a depth image
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        
        # Remove background
        depth_image = removeBackground(depth_image)
        
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
        
        # Draw images side by side
        images = np.hstack((depth_colormap, contour_map))
        cv.namedWindow('Contour Example', cv.WINDOW_AUTOSIZE)
        cv.imshow('Contour Example', images)
        key = cv.waitKey(1)

    pipeline.stop()

if __name__ == '__main__':
    main();