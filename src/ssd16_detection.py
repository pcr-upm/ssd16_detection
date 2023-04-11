#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import cv2
from images_framework.src.annotations import FaceObject, GenericCategory
from images_framework.src.detection import Detection
from images_framework.src.categories import Category as Oi


class SSD16Detection(Detection):
    """
    Object detection using SSD algorithm
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.gpu = None

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='SSD16Detection', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.gpu = args.gpu

    def train(self, anns_train, anns_valid):
        import caffe
        print('CPU mode' if -1 in self.gpu else 'GPU mode with devices ' + str(self.gpu))
        if -1 in self.gpu:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
        # Create Caffe neural network
        print('Running solver ...')
        solver_file = self.path + 'data/ResNet_152_solver.prototxt'
        solver = caffe.get_solver(solver_file)
        # Train from scratch vs Finetune from a previous net
        model_file = self.path + 'data/ResNet_152.caffemodel'
        if os.path.exists(model_file):
            solver.net.copy_from(model_file)
        solver.solve()

    def load(self, mode):
        from images_framework.src.constants import Modes
        # Set up a neural network to train
        print('Load model')
        proto_file = self.path + 'data/deploy.prototxt'
        model_file = self.path + 'data/res10_300x300_ssd_iter_140000.caffemodel'
        if mode is Modes.TEST:
            self.model = cv2.dnn.readNetFromCaffe(proto_file, model_file)

    def process(self, ann, pred):
        for img_pred in pred.images:
            # Load image
            image = cv2.imread(img_pred.filename)
            input_blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
            self.model.setInput(input_blob, 'data')
            output = self.model.forward()
            # Save prediction
            for detection in output[0][0]:
                if detection[2] < 0.5:
                    continue
                obj = FaceObject()
                obj.bb = detection[3:7] * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
                obj.add_category(GenericCategory(label=Oi.FACE, score=detection[2]))
                img_pred.add_object(obj)
