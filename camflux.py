##
## EPITECH PROJECT, 2025
## camrobocar
## File description:
## main
##

import depthai as dai
import cv2
from ray_generator import generate_rays, show_rays

def tmp_mask(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * 0

pipeline = dai.Pipeline()

cam_color = pipeline.createColorCamera()
cam_color.setPreviewSize(640, 480)
cam_color.setInterleaved(False)
cam_color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_color.preview.link(xout.input)

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    while True:
        in_video = video_queue.get()
        frame = in_video.getCvFrame()

        cv2.imshow("Color Camera", frame)
        mask = tmp_mask(frame)
        cv2.imshow("Mask", mask)
        # convert mask to grayscale
        distances, ray_endpoints = generate_rays(mask, num_rays=50, fov_degrees=120, max_distance=100)
        show_rays(mask, ray_endpoints)
        if cv2.waitKey(1) == ord('q'):
            break


