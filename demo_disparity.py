from pipelime.sequences.readers.filesystem import UnderfolderReader
import time
import torch.nn.functional as F
import kornia.geometry as geometry
import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

import depthai as dai
import depthai

import threading


class OAKSensor:

    def __init__(self) -> None:
        pipeline = dai.Pipeline()

        # RGB Camera
        self.rgb_camera = pipeline.createColorCamera()
        self.rgb_camera.setPreviewSize(640, 400)
        self.rgb_camera.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.rgb_camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.rgb_camera.setInterleaved(False)
        self.rgb_camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Left camera
        self.left_camera = pipeline.createMonoCamera()
        self.left_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.left_camera.setBoardSocket(dai.CameraBoardSocket.LEFT)

        # Right camera
        self.right_camera = pipeline.createMonoCamera()
        self.right_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.right_camera.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Stereo camera
        self.stereo_camera = pipeline.createStereoDepth()
        self.stereo_camera.setConfidenceThreshold(255)
        self.stereo_camera.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
        self.stereo_camera.setLeftRightCheck(True)
        self.stereo_camera.setExtendedDisparity(False)
        self.stereo_camera.setSubpixel(False)
        # stereo_camera.setOutputDepth(True)
        # self.stereo_camera.setOutputRectified(True)

        # Link LEFT/RIGHT -> STEREO
        self.left_camera.out.link(self.stereo_camera.left)
        self.right_camera.out.link(self.stereo_camera.right)

        # RGB Stream
        self.rgb_stream = pipeline.createXLinkOut()
        self.rgb_stream.setStreamName("rgb_stream")
        self.rgb_camera.preview.link(self.rgb_stream.input)

        # Left Stream
        self.left_stream = pipeline.createXLinkOut()
        self.left_stream.setStreamName("left_stream")
        self.stereo_camera.rectifiedLeft.link(self.left_stream.input)
        # stereo_camera.rectifiedLeft.link(left_stream.input)

        # Right Stream
        self.right_stream = pipeline.createXLinkOut()
        self.right_stream.setStreamName("right_stream")
        self.stereo_camera.rectifiedRight.link(self.right_stream.input)

        # Depth Stream
        self.depth_stream = pipeline.createXLinkOut()
        self.depth_stream.setStreamName("depth_stream")
        self.stereo_camera.depth.link(self.depth_stream.input)

        # Camera control
        self.camControlIn = pipeline.createXLinkIn()
        self.camControlIn.setStreamName('camControl')
        self.camControlIn.out.link(self.rgb_camera.inputControl)

        # Device
        self.device = dai.Device(pipeline, usb2Mode=False)
        self.device.startPipeline()
        self.device.setLogLevel(depthai.LogLevel.DEBUG)
        # device.setLogOutputLevel(depthai.LogLevel.DEBUG)
        # device.getQueueEvents()

        self.streams = {
            'rgb_stream': self.device.getOutputQueue(name="rgb_stream", maxSize=1, blocking=True),
            'left_stream': self.device.getOutputQueue(name="left_stream", maxSize=1, blocking=True),
            'right_stream': self.device.getOutputQueue(name="right_stream", maxSize=1, blocking=True),
            'depth_stream': self.device.getOutputQueue(name="depth_stream", maxSize=1, blocking=True),
        }

        # Set Focus
        # Send controls to camera

        current_focus = 130
        self.set_focus(current_focus)

        self._buffer = {}
        self.running = True
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def grab(self):
        if all([x in self._buffer for x in self.streams.keys()]):
            new_buffer = {}
            valid = True
            for k, v in self._buffer.items():
                if v is not None:
                    new_buffer[k] = v.getCvFrame().copy()
                else:
                    valid = False
                    break
            if valid:
                return new_buffer
        return None

    def loop(self):

        # Loop
        while self.running:
            # print("New_frame")

            for name in self.streams.keys():
                frame = self.device.getOutputQueue(name).tryGet()
                if frame is not None:
                    self._buffer[name] = frame
                time.sleep(0.002)

    def set_focus(self, focus):
        qControl = self.device.getInputQueue(name="camControl")
        camControl = dai.CameraControl()
        camControl.setAutoFocusMode(depthai.RawCameraControl.AutoFocusMode.OFF)
        camControl.setManualFocus(focus)
        qControl.send(camControl)
        print("Focus:", focus)


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img)
    if len(img.shape) == 2:
        img = img.unsqueeze(2).repeat(1, 1, 3)
    img = img.permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def view(img, name):
    img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img_flo[:, :, [2, 1, 0]]/255.0)
    cv2.waitKey()


MAX_DEPTH = 2


def viz_gt(depth):
    depth = depth / 1000
    if MAX_DEPTH > 0:
        depth[depth < 0] = 0
        depth[depth > MAX_DEPTH] = 0
        depth = (255 * depth / MAX_DEPTH).astype(np.uint8)
    else:
        depth = cv2.normalize(depth, depth, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)

    print("DEPTH"*10, depth.min(), depth.max())
    cv2.imshow('depth_gt', depth)


def viz_disparity(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    disp = np.abs(flo[:, :, 0])

    depth = 0.075 * 414.9395794176142 / disp
    if MAX_DEPTH > 0:
        depth[depth < 0] = 0
        depth[depth > MAX_DEPTH] = 0
        depth = (255 * depth / MAX_DEPTH).astype(np.uint8)
    else:
        depth = cv2.normalize(depth, depth, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    # img_flo = np.concatenate([img, disp], axis=0)

    # # import matplotlib.pyplot as plt
    # # plt.imshow(img_flo / 255.0)
    # # plt.show()
    print(disp.min(), disp.max(), disp.shape)
    print(depth.min(), depth.max(), depth.shape)

    # disp = cv2.applyColorMap(disp.astype(np.uint8), cv2.COLORMAP_MAGMA)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img.astype(np.uint8))
    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.imshow('disp', disp.astype(np.uint8))
    cv2.imshow('depth', depth)

    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    dataset = UnderfolderReader(folder='/tmp/hammerboy_dataset')

    with torch.no_grad():
        for sample in dataset:

            im1 = sample['right']
            im2 = sample['left']

            image1 = torch.Tensor(im1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(DEVICE)
            image2 = torch.Tensor(im2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(DEVICE)

            t1 = time.perf_counter()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            t2 = time.perf_counter()
            print("Time: ", t2 - t1)

            # view(image1_warped, 'img2_warped')
            # view(image1, 'img2')
            viz_gt(sample['depth'])
            viz_disparity(image1, flow_up)
            cv2.waitKey(0)


def demo2(args):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    sensor = OAKSensor()
    print("Sensor ok")

    with torch.no_grad():
        while True:
            buffer = sensor.grab()
            if buffer:
                im1 = cv2.flip(buffer['left_stream'], 1)
                im2 = cv2.flip(buffer['right_stream'], 1)

                image1 = torch.Tensor(im1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(DEVICE)
                image2 = torch.Tensor(im2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(DEVICE)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                t1 = time.perf_counter()
                flow_low, flow_up = model(image1, image2, iters=30, test_mode=True)

                t2 = time.perf_counter()
                print("Time: ", t2 - t1)

                # view(image1_warped, 'img2_warped')
                # view(image1, 'img2')
                viz_disparity(image1, flow_up)

                viz_gt(buffer['depth_stream'])
                # cv2.imshow("image", buffer['rgb_stream'])
                cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
