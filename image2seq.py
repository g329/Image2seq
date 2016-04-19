#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
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

#sys.stdout = codecs.getwriter('utf_8')(sys.stdout)


class Image2Seq(chainer.Chain):
    def __init__(self, output_vocab, feature_num, hidden_num):
        """

        :param input_image:  image (3,227,227)
        :param output_vocab: array of output space vocab
        :param feature_num: size of feature layer
        :param hidden_num: size of hidden layer
        :return:
        """

        self.dropout_ratio = 0.5
        self.feature = FeatureExtractor("./model/feature_alexbn.npz")
        # TODO TO GPU
        # self.feature.model.to_gpu()


        self.id2word_output = {}
        self.word2id_output = {}
        for id, word in enumerate(output_vocab):
            self.id2word_output[id] = word
            self.word2id_output[word] = id

        self.output_vocab_size = len(self.word2id_output)
        print "vocabs : ", self.output_vocab_size

        super(Image2Seq, self).__init__(

                # connect layer
                context_lstm=L.LSTM(9216, self.output_vocab_size),

                # decoder
                output_lstm_1=L.LSTM(self.output_vocab_size, hidden_num),
                output_lstm_2=L.LSTM(hidden_num, hidden_num),
                output_lstm_3=L.LSTM(hidden_num, feature_num),
                out_word=L.Linear(feature_num, self.output_vocab_size),
        )

    def get_feature(self,src_images):
        feature = self.feature(src_images)
        return feature.data

    # TODO image encoder
    def encode(self, feature, train=True):
        """

        :param : feature xp
        :return: context vector
        """
        feature = chainer.Variable(feature)
        context = self.context_lstm(feature)

        return context

    def decode(self, context, teacher_embed_id, train):
        """

        :param context: context vector which made `encode` function
        :return: decoded embed vector
        """

        output_feature = self.output_lstm_1(context )
        output_feature = self.output_lstm_2(output_feature)
        output_feature = self.output_lstm_3(output_feature)

        predict_embed_id = self.out_word(output_feature)
        if train:
            t = xp.array([teacher_embed_id], dtype=xp.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_embed_id, t), predict_embed_id
        else:
            return predict_embed_id

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
        self.output_lstm_3.reset_state()

    def generate(self, feature, sentence_limit, gpu=-1):
        """

        :param feature: image feature
        :param sentence_limit:
        :param gpu:
        :return:
        """

        sentence = ""
        length = 0
        context = self.encode(feature)

        while length < sentence_limit:
            decoded_feature = self.decode(context, teacher_embed_id=None, train=False)

            if gpu >= 0:
                word_id = cuda.to_cpu(decoded_feature.data)
            else:
                word_id = decoded_feature.data
            context = decoded_feature

            word = self.id2word_output[np.argmax(word_id)]
            if word == "<end>":
                break
            sentence = sentence + word + " "
            length += 1
        return sentence


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1702)
    args = parser.parse_args()
    np.random.seed(args.seed)

    # datas : dictionary
    # datas["file_name"] : 入力画像
    # datas["text"] : 目標sequence
    datas = pickle.load(open("./habomaijiro.pkl", "rb"))
    datas = datas[:3]

    loader = ImageLoader(size=(227, 227), mean=None)

    xp = cuda.cupy if args.gpu >= 0 else np

    # URLを除去
    words = [re.sub('(http|https)://[A-Za-z0-9\'~+\-=_.,/%\?!;:@#\*&\(\)]+', " ", data["text"]) for data in datas]
    output_vocab = make_vocab(words)

    model = Image2Seq(output_vocab, feature_num=50, hidden_num=100)
    if args.gpu >= 0 :
        model.to_gpu()

    optimizer = optimizers.Adam()
    # optimizer = optimizers.SGD()
    optimizer.setup(model)

    # path 2 images(1.jpg , 2.jpg ...)
    path = "./data/images/"
    images = [loader.load(path + data["file_name"]) for data in datas]

    # first run
    features = [model.get_feature(xp.array([image], dtype=np.float32)) for image in images]
    pickle.dump(features,open("./features.npy","wb"),-1)
    # second run
    # features = pickle.load(open("./features.npy", "r"))

    # datas["text"] : 分かち書き済みtext
    # datas["file_name"] : 画像ファイル名
    # datas["feature"] : 特徴ベクトル
    datas = [{"text": data["text"],
              "wakati_text": wakati(re.sub('(http|https)://[A-Za-z0-9\'~+\-=_.,/%\?!;:@#\*&\(\)]+', " ", data["text"])),
              "file_name": path + data["file_name"], "feature": features[i]} for i, data in enumerate(datas)]

    for epoch in range(100000):
        acc_loss = 0
        # すべてのデータを読み込む
        for i, data in enumerate(datas):

            model.initialize()

            # image -> feature
            feature = data["feature"]
            context = model.encode(feature,train=True)
            teacher_text = data["wakati_text"]

            # TODO datas[63] and datas[352] has unexpected char code
            try:
                for word in teacher_text:
                    id = model.word2id_output[word]
                    loss, context = model.decode(context, id, train=True)
                    acc_loss += loss
            except:
                # if catch char code error
                # print "data[%d] " % i ,  " has unexpected char code"
                continue

            model.zerograds()
            acc_loss.backward()
            acc_loss.unchain_backward()
            optimizer.update()
            acc_loss = 0
            model.initialize()

        test_context = datas[epoch % len(datas)]["feature"]
        sentence = model.generate(test_context, 100, args.gpu)
        answer = datas[epoch % len(datas)]["text"]

        print "(%d) epoch " % epoch
        print "*** answer ***\n", answer, "\n"
        print "*** generated *** \n", sentence, "\n"
        print "==================\n\n"

        if epoch % 100 == 0:
            pass
            #serializers.save_npz("model_%d.npz" % epoch, model)
            #print "model_%d.npz saved" % epoch
