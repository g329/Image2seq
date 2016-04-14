#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
from chainer.functions import caffe
import pickle
from chainer import serializers

import chainer
import chainer.functions as F
import chainer.links as L


class FeatureAlexBN(chainer.Chain):
    """Single-GPU AlexNet with LRN layers replaced by BatchNormalization."""

    insize = 227

    def __init__(self):
        super(FeatureAlexBN, self).__init__(
                conv1=L.Convolution2D(3, 96, 11, stride=4),
                bn1=L.BatchNormalization(96),
                conv2=L.Convolution2D(96, 256, 5, pad=2),
                bn2=L.BatchNormalization(256),
                conv3=L.Convolution2D(256, 384, 3, pad=1),
                conv4=L.Convolution2D(384, 384, 3, pad=1),
                conv5=L.Convolution2D(384, 256, 3, pad=1),
                fc6=L.Linear(9216, 4096),
                fc7=L.Linear(4096, 4096),
                fc8=L.Linear(4096, 1000),
        )
        self.train = True

    def __call__(self, x):
        x = chainer.Variable(x)
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.bn2(self.conv2(h), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        # h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        # h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        # h = self.fc8(h)

        # self.loss = F.softmax_cross_entropy(h, t)
        # self.accuracy = F.accuracy(h, t)
        return h


if __name__ == "__main__":

    # 特徴抽出機
    model_net = FeatureAlexBN()
    caffe_alex = caffe.CaffeFunction("./bvlc_alexnet.caffemodel")
    pickle.dump(caffe_alex, open("./alexnet_bn.pkl","wb"))
    # vgg = pickle.load(open("./VGG16.caffepickle"))
    layers = [layer_name for layer_name in dir(caffe_alex) if ("conv" in layer_name) and (not "setup" in layer_name)]

    print "caffemodel copy start"
    for layer_name in layers:
        layer = getattr(caffe_alex, layer_name)
        setattr(model_net, layer_name, layer)

    # copy 確認
    serializers.save_npz("./feature_alexbn.npz", model_net)
    print "caffemodel copy done"
    print "if u want to use this model , "
    print "vgg_net = VGGNet() "
    print 'serializers.load_hdf5("filename",vgg_net) '

    """
        caffe attr names
        conv1_1 (64, 3, 3, 3)
        conv1_2 (64, 64, 3, 3)
        conv2_1 (128, 64, 3, 3)
        conv2_2 (128, 128, 3, 3)
        conv3_1 (256, 128, 3, 3)
        conv3_2 (256, 256, 3, 3)
        conv3_3 (256, 256, 3, 3)
        conv4_1 (512, 256, 3, 3)
        conv4_2 (512, 512, 3, 3)
        conv4_3 (512, 512, 3, 3)
        conv5_1 (512, 512, 3, 3)
        conv5_2 (512, 512, 3, 3)
        conv5_3 (512, 512, 3, 3)
        fc6 (4096, 25088)
        fc7 (4096, 4096)
        fc8-5 (5, 4096)
    """
