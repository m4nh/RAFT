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


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
            glob.glob(os.path.join(args.path, '*.jpg')) + \
            glob.glob(os.path.join(args.path, '*.tiff'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            t1 = time.perf_counter()
            flow_low, flow_up = model(image1, image2, iters=5, test_mode=True)
            t2 = time.perf_counter()
            print("Time: ", t2 - t1)
            ############
            height, width = image1.shape[-2:]
            grid = geometry.create_meshgrid(height, width, normalized_coordinates=False).to(image1.device)
            print("SPODSAOPDA", flow_up.shape, grid.shape, grid.min(), grid.max())
            grid = flow_up.permute(0, 2, 3, 1) + grid
            flow_up_norm = geometry.normalize_pixel_coordinates(grid, height, width)  # BxHxWx2

            image1_warped = F.grid_sample(image2, flow_up_norm, align_corners=True)

            view(image1_warped, 'img2_warped')
            view(image1, 'img2')
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
