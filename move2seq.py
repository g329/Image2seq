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


class Movie2Seq(chainer.Chain):
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

        super(Movie2Seq, self).__init__(

                # connect layer
                context_lstm=L.LSTM(9216, feature_size),
                previous_h=L.Linear(class_num, hidden_size),

                # decoder
                output_lstm_1=L.LSTM(feature_size + 3, hidden_size),
                output_lstm_2=L.LSTM(hidden_size, hidden_size),
                l_3=L.Linear(hidden_size, class_num),
        )

    def __call__(self, flame, t, previous_h=None, train=False):
        """

        return one-flame loss and h
        :param x: Movie (Numpy or Cupy Array)
        :param t: Vector
        :return:
        """
        print np.shape(flame),np.shape(t)
        t = chainer.Variable(t)

        feature = self.feature(np.array([flame],dtype=np.float32))
        h = self.context_lstm(feature)

        # default previous_h is Zero
        if previous_h is None:
            previous_h = chainer.Variable(np.zeros((1, class_num), dtype=np.float32))

        h = F.concat((h, previous_h))

        h = F.dropout(h, ratio=0.5, train=train)
        h = self.output_lstm_1(h)
        h = self.output_lstm_2(h)
        h = self.l_3(h)

        if train:
            # loss and previous h
            return F.softmax_cross_entropy(h, t), F.softmax(h)
        else:
            return F.softmax(h)


    def foward(self, x, t):
        pass


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
    parser.add_argument('--epoch', type=int, default=1)
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

    class_num = len(datas.keys())
    model = Movie2Seq(class_num, 100, 100)

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # TODO 学習データとテストデータに分ける

    # input one sequence
    loss, previous_h = model(datas["clap"]["datas"][0][0], datas["clap"]["labels"], previous_h=None, train=True)
    print loss.data, previous_h.data
    loss, previous_h = model(datas["clap"]["datas"][0][0], datas["clap"]["labels"], previous_h=previous_h, train=True)
    print loss.data, previous_h.data
    exit()

    for epoch in range(args.epoch):
        pass

    model.initialize()

    model.zerograds()
    loss.backward()
    loss.unchain_backward()
    optimizer.update()
    loss = 0
    model.initialize()





    # TODO 学習部分
    # TODO 結果吐き出し
