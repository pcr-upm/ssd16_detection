#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import cv2
from images_framework.src.annotations import PersonObject, FaceObject, GenericCategory
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
        self.width = 300
        self.height = 300
        self.filename = None
        self.category = None
        self.mean = None
        self.std = None

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='SSD16Detection', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.gpu = args.gpu
        if self.database in ['coco', 'pascal_voc']:
            # The model was created with SSD framework using MobileNet like architecture as a backbone.
            # The model was first trained on the COCO dataset and was then fine-tuned on PASCAL VOC.
            self.filename = 'mobilenet'
            self.category = Oi.PERSON
            self.mean = (127.5, 127.5, 127.5)
            self.std = 0.007843
        elif self.database in ['300w_public', '300w_private', 'cofw', 'aflw', 'wflw', 'dad', '300wlp']:
            # The model was created with SSD framework using ResNet-10 like architecture as a backbone.
            # The model was trained in Caffe framework on some huge and available online dataset.
            self.filename = 'res10_300x300_ssd'
            self.category = Oi.FACE
            self.mean = (104.0, 177.0, 123.0)
            self.std = 1.0
        else:
            raise ValueError('Database is not implemented')

    def train(self, anns_train, anns_valid):
        import caffe
        print('CPU mode' if -1 in self.gpu else 'GPU mode with devices ' + str(self.gpu))
        if -1 in self.gpu:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
        # Create Caffe neural network
        print('Running solver ...')
        solver_file = self.path + 'data/' + self.category.label.name + '/' + 'solver.prototxt'
        solver = caffe.get_solver(solver_file)
        # Train from scratch vs Finetune from a previous net
        model_file = self.path + 'data/' + self.category.label.name + '/' + self.filename + '.caffemodel'
        if os.path.exists(model_file):
            solver.net.copy_from(model_file)
        solver.solve()

    def load(self, mode):
        from images_framework.src.constants import Modes
        # Set up a neural network to train
        print('Load model')
        proto_file = self.path + 'data/' + self.category.name + '/' + 'deploy.prototxt'
        model_file = self.path + 'data/' + self.category.name + '/' + self.filename + '.caffemodel'
        if mode is Modes.TEST:
            self.model = cv2.dnn.readNetFromCaffe(proto_file, model_file)

    def process(self, ann, pred):
        for img_pred in pred.images:
            # Load image
            image = cv2.imread(img_pred.filename)
            input_blob = cv2.dnn.blobFromImage(image, scalefactor=self.std, size=(self.width, self.height), mean=self.mean, swapRB=False, crop=False)
            self.model.setInput(input_blob, 'data')
            output = self.model.forward()
            # Save prediction
            for detection in output[0][0]:
                if detection[2] < 0.5:
                    continue
                obj = PersonObject() if self.category == Oi.PERSON else FaceObject()
                obj.bb = tuple(detection[3:7] * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                obj.add_category(GenericCategory(label=self.category, score=detection[2]))
                img_pred.add_object(obj)
