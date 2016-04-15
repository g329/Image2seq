#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import argparse
import os
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


class Image2Seq(chainer.Chain):
    def __init__(self, class_num, feature_size, hidden_size):
        """

        :param input_image:  image (3,227,227)
        :param output_vocab: array of output space vocab
        :param feature_num: size of feature layer
        :param hidden_num: size of hidden layer
        :return:
        """

        self.dropout_ratio = 0.5
        self.feature = FeatureExtractor("./model/feature_alexbn.npz")

        super(Image2Seq, self).__init__(

                # connect layer
                context_lstm=L.LSTM(9216, feature_size),
                previous_h=L.Linear(class_num, hidden_size),

                # decoder
                output_lstm_1=L.LSTM(feature_size, hidden_size),
                output_lstm_2=L.LSTM(hidden_size, hidden_size),
                l_3=L.Linear(hidden_size, class_num),
        )

    # TODO image encoder
    def encode(self, src_images, train=True):
        """
        時系列情報を抽出
        :param src_images : chainer Variable , (N,3,227,227)
        :return: context vector
        """

        for _ in range(src_images.data):
            feature = self.feature(src_images)
            context = self.context_lstm(feature)

        return context

    def decode(self, context, previous_h, t, train):
        """

        :param context: context vector which made `encode` function
        :return: decoded embed vector
        """
        data = F.concat((context, previous_h))

        # output_feature = self.output_lstm_1(F.dropout(data, ratio=self.dropout_ratio, train=train))
        # output_feature = self.output_lstm_2(F.dropout(output_feature, ratio=self.dropout_ratio, train=train))
        output_feature = self.output_lstm_1(data)
        output_feature = self.output_lstm_2(output_feature)
        h = self.l_3(output_feature)

        if train:
            return F.softmax_cross_entropy(h, t), F.softmax(h)
        else:
            return F.softmax(h)

    def initialize(self):
        """
        init state


        :param image_feature:
        :param train:
        :return:
        """
        self.context_lstm.reset_state()
        self.output_lstm_1.reset_state()
        self.output_lstm_2.reset_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1702)
    args = parser.parse_args()
    np.random.seed(args.seed)

    xp = cuda.cupy if args.gpu >= 0 else np

    # datas : dictionary
    # datas["file_name"] : 入力動画の
    # datas["text"] : 目標sequence

    datas = {}
    for file_name in [file_name for file_name in os.listdir("./data/movies") if ".pkl" in file_name]:
        datas[file_name.split("_")[0]] = pickle.load(open("./data/movies/clap_30F.pkl", "r"))

    class2id = pickle.load(open("./data/movies/class2id.pkl", "r"))
    id2class = {class2id[class_name]: class_name for class_name in class2id.keys()}

    print class2id
    print id2class

    exit()

    class_num = len(datas.keys())

    # TODO 学習データとテストデータに分ける

    for epoch in range(args.epoch):
        pass
        # TODO 学習部分
        # TODO 結果吐き出し
