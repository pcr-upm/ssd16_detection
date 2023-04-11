#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import json
import copy
import argparse
import numpy as np
from pathlib import Path
from images_framework.src.constants import Modes
from images_framework.src.composite import Composite
from images_framework.detection.ssd16_detection.src.ssd16_detection import SSD16Detection
from images_framework.src.annotations import GenericGroup, GenericImage, FaceObject
from images_framework.src.viewer import Viewer
from images_framework.src.utils import load_geoimage

image_extensions = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff')
video_extensions = ('mp4', 'avi', 'mkv')


def parse_options():
    """
    Parse options from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', '-d', dest='input_data', required=False, default='',
                        help='Input as image, directory, camera or video file.')
    parser.add_argument('--show-viewer', '-v', dest='show_viewer', action="store_true",
                        help='Show results visually.')
    parser.add_argument('--save-image', '-i', dest='save_image', action="store_true",
                        help='Save processed images.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    input_data = args.input_data
    show_viewer = args.show_viewer
    save_image = args.save_image
    return unknown, input_data, show_viewer, save_image


def process_frame(composite, filename, show_viewer, save_image, viewer, delay, dirname):
    """
    Process frame and show results.
    """
    ann = GenericGroup()
    pred = GenericGroup()
    img_pred = GenericImage(filename)
    img, _ = load_geoimage(img_pred.filename)
    img_pred.tile = np.array([0, 0, img.shape[1], img.shape[0]])
    ann.add_image(copy.deepcopy(img_pred))
    # Read annotations from a json file
    ann_filename = os.path.splitext(filename)[0]+'.json'
    if os.path.exists(ann_filename):
        for ann_json in json.load(open(ann_filename))['annotations']:
            obj = FaceObject()
            x, y, w, h = ann_json['bbox']
            obj.bb = (x, y, x+w, y+h)
            img_pred.add_object(obj)
    pred.add_image(img_pred)
    ticks = cv2.getTickCount()
    composite.process(ann, pred)
    ticks = cv2.getTickCount() - ticks
    if show_viewer:
        for img_pred in pred.images:
            viewer.set_image(img_pred)
        composite.show(viewer, ann, pred)
        fps = 'FPS = ' + "{0:.3f}".format(cv2.getTickFrequency() / ticks)
        viewer.text(pred.images[0], fps, (20, np.shape(viewer.get_image(pred.images[0]))[0] - 20), 0.5, (0, 255, 0))
        viewer.show(delay)
    if save_image:
        for img_pred in pred.images:
            viewer.set_image(img_pred)
        composite.show(viewer, ann, pred)
        viewer.save(dirname + os.path.basename(pred.images[0].filename))
        composite.save(dirname, pred)


def main():
    """
    SSD 2016 test script.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, input_data, show_viewer, save_image = parse_options()

    # Determine if we get the images from a camera, video or directory
    process_image, process_directory, process_video = False, False, False
    if os.path.isfile(input_data):
        if input_data.lower().endswith(image_extensions):
            print('Processing an image file ...')
            process_image = True
        elif input_data.lower().endswith(video_extensions):
            print('Processing from a video file ...')
            process_video = True
            capture = cv2.VideoCapture(input_data)
            if not capture.isOpened():
                raise ValueError('Could not grab images from video')
        else:
            raise ValueError('Unknown input file extension')
    else:
        if os.path.isdir(input_data):
            print('Processing a directory ...')
            process_directory = True
        else:
            print('Processing from camera ...')
            process_video = True
            capture = cv2.VideoCapture(0)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not capture.isOpened():
                raise ValueError('Could not grab images from camera')

    # Load computer vision components
    composite = Composite()
    sd = SSD16Detection('images_framework/detection/ssd16_detection/')
    composite.add(sd)
    composite.parse_options(unknown)
    composite.load(Modes.TEST)
    viewer = Viewer('ssd16_detection_test')
    dirname = 'output/images/'
    Path(dirname).mkdir(parents=True, exist_ok=True)

    # Process frame and show results
    if process_image:
        process_frame(composite, input_data, show_viewer, save_image, viewer, 0, dirname)
    elif process_directory:
        for filename in os.listdir(input_data):
            filename = input_data + '/' + filename
            if not filename.lower().endswith(image_extensions):
                continue
            process_frame(composite, filename, show_viewer, save_image, viewer, 0, dirname)
    elif process_video:
        idx = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            # Save camera frame into disk to continue processing
            filename = dirname + str(idx).zfill(5) + '.png'
            cv2.imwrite(filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            idx += 1
            process_frame(composite, filename, show_viewer, save_image, viewer, 1, dirname)
        capture.release()
    print('End of ssd16_detection_test')


if __name__ == '__main__':
    main()
