#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
from chainer.functions import caffe
import pickle
from chainer import serializers
from tools import io_util
import numpy as np
import chainer
from model import AlexNet
import chainer.functions as F
import chainer.links as L


class FeatureExtractor():
    def __init__(self, feature_pkl_path="../model/feature_alexbn.npz"):
        model = AlexNet.FeatureAlexBN()
        chainer.serializers.load_npz(feature_pkl_path, model)
        self.model = model

    def __call__(self, images):
        """
        :param image: image.shape = (1,3,227,227) alex net shape
        :return:
        """
        return self.model(images)


if __name__ == "__main__":
    loader = io_util.ImageLoader(size=(227, 227), mean=None)
    image = loader.load("../data/0.jpg")
    data = np.array([image,image], dtype=np.float32)
    feature = FeatureExtractor()

    print " --- feature --- "
    print feature(data).data.shape

