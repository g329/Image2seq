#!/usr/bin/env python
# -*- coding:utf-8 -*-

from skimage import io
import chainer
import chainer.functions as F
import chainer.links as L
import itertools
import argparse
from chainer import serializers
from chainer import optimizers
import numpy as np
from chainer import cuda
import sys
import codecs
import pickle
import re
from tools.feature_extractor import FeatureExtractor
from tools.io_util import ImageLoader
from tools.wakater import make_vocab
from tools.wakater import wakati


def make_patch(image, patch_size=(3, 9, 9)):
    """

    :param image: 3,H,W (numpy.array)
    :param patch_size:
    :return: patches[(30,30)] = pixel
    """

    image_shape = image.shape
    dots = itertools.product(range(image_shape[1]), range(image_shape[2]))

    p_width = (patch_size[1]) / 2
    p_height = (patch_size[2]) / 2

    patches = {}

    for points in dots:
        x, y = points[0:2]
        pixel = src_image[:, x - p_width: x + p_width + 1, y - p_height: y + p_height + 1]

        # ハミ出るなどして，9x9にならなかったものは無視する
        if pixel.shape != patch_size:
            continue
        else:
            patches[points] = pixel

    return patches


class SuperFlame(chainer.Chain):
    """
    周囲13フレームから超解像ピクセルを推定するモデル
    """

    def __init__(self):
        super(SuperFlame, self).__init__(
                conv1=L.Convolution2D(3, 64, ksize=3, stride=1, pad=1),
                conv2=L.Convolution2D(64, 128, ksize=9, stride=1, pad=0),
                conv3=L.Convolution2D(128, 64, ksize=1, stride=1, pad=0),
                conv4=L.Convolution2D(64, 3, ksize=1, stride=1, pad=0)
        )

    def __call__(self, src_patches, dst_image, train, patch_size=(3, 9, 9)):
        """
        超解像化した画像を返す
        :param src_image:
        :param dst_image:
        :param train:
        :return:
        """
        super_resolution = xp.zeros(xp.shape(dst_image), dtype=xp.float32)
        acc_loss = 0
        for pixel,patch in src_patches.items():
            # for fix data shape
            t = xp.zeros((1,3,1,1),dtype=xp.float32)
            for i,elem in enumerate(dst_image[:,pixel[0],pixel[1]]):
                t[0][i][0][0] = elem

            patch = xp.array([patch] , dtype=xp.float32)
            loss = model.foward(patch,t,train=train)
            acc_loss += loss

        return acc_loss


    def foward(self, src_pixel, dst_pixel, train):
        """

        :param src_pixel: 3x9x9
        :param dst_pixel: 3x1x1
        :param train:
        :return:
        """

        x = chainer.Variable(src_pixel)
        t = chainer.Variable(dst_pixel)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)

        if train:
            return F.mean_squared_error(h, t)
        else:
            return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1702)
    args = parser.parse_args()
    np.random.seed(args.seed)

    # loader = ImageLoader(size=(227, 227), mean=None)
    path = "../data/images/super_flame/"
    # src_image = loader.load(path + "arai.png")
    src_image = io.imread(path + "arai.png").transpose((2, 0, 1))
    image_shape = src_image.shape

    target_image = io.imread(path + "kirei.png").transpose((2, 0, 1))

    # HOW TO SHOW
    # src_image = src_image.transpose((1, 2, 0))
    # io.imshow(src_image)
    # io.show()

    patch_size = (9, 9, 3)
    xp = cuda.cupy if args.gpu >= 0 else np

    model = SuperFlame()
    if args.gpu >= 0 :
        model.to_gpu(args.gpu)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # HOW TO MAKE DATA_SET
    # arai = make_patch(src_image)
    # pickle.dump(arai, open("./arai_patch_9.pkl","wb"), -1)
    # kirei = make_patch(target_image)
    # pickle.dump(kirei, open("./kirei_patch_9.pkl","wb"), -1)
    arai_patch = pickle.load(open("./arai_patch_9.pkl", "r"))



    for _ in range(args.epoch):
        acc_loss = model(arai_patch, target_image, train=True)
        print acc_loss.data
        model.zerograds()
        acc_loss.backward()
        acc_loss.unchain_backward()
        optimizer.update()
        acc_loss = 0
