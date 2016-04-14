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

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)


class Image2Seq(chainer.Chain):
    dropout_ratio = 0.5

    def __init__(self, output_vocab, feature_num, hidden_num):
        """

        :param input_image:  image (3,227,227)
        :param output_vocab: array of output space vocab
        :param feature_num: size of feature layer
        :param hidden_num: size of hidden layer
        :return:
        """
        dict = {}
        self.feature = FeatureExtractor("./model/feature_alexbn.npz")

        self.id2word_output = {}
        self.word2id_output = {}
        for id, word in enumerate(output_vocab):
            self.id2word_output[id] = word
            self.word2id_output[word] = id

        self.output_vocab_size = len(self.word2id_output)
        print "vocabs : ", self.output_vocab_size

        super(Image2Seq, self).__init__(
                # encoder
                # word_vec=L.EmbedID(self.input_vocab_size, feature_num),
                # TODO image encode layer
                # input_vec=L.Linear(feature_num, hidden_num),
                # input_vec=L.LSTM(feature_num, hidden_num),

                # connect layer
                context_lstm=L.LSTM(9216, hidden_num),

                # decoder
                output_lstm_1=L.LSTM(hidden_num, feature_num),
                output_lstm_2=L.LSTM(feature_num, feature_num),
                output_lstm_3=L.LSTM(feature_num, feature_num),
                out_word=L.Linear(feature_num, self.output_vocab_size),
        )

    # TODO image encoder
    def encode(self, src_images, train=True):
        """

        :param src_images : chainer Variable , (N,3,227,227)
        :return: context vector
        """

        feature = self.feature(src_images)
        context = self.context_lstm(feature)

        return context

    def decode(self, context, teacher_embed_id, train):
        """

        :param context: context vector which made `encode` function
        :return: decoded embed vector
        """

        output_feature = self.output_lstm_1(F.dropout(context, ratio=self.dropout_ratio, train=train))
        output_feature = self.output_lstm_2(F.dropout(output_feature, ratio=self.dropout_ratio, train=train))
        output_feature = self.output_lstm_3(F.dropout(output_feature, ratio=self.dropout_ratio, train=train))

        predict_embed_id = self.out_word(output_feature)
        if train:
            t = xp.zeros(1, dtype=xp.int32)
            t[0] = teacher_embed_id
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_embed_id, t)
        else:
            return predict_embed_id

    def initialize(self):  # train=True):
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
        # self.lstm.reset_state()
        # h = self.image_vec(F.dropout(image_feature, ratio=self.dropout_ratio, train=train))
        # self.lstm(F.dropout(h, ratio=self.dropout_ratio, train=train))

    def __call__(self, word, train=True):
        """
        :param word:
        :param train:
        :return:
        """

        # TODO train code

        embed_vector = self.word_vec(word)
        input_feature = self.input_vec(embed_vector)
        connector = self.connect_lstm(F.dropout(input_feature, ratio=self.dropout_ratio, train=train))
        output_feature = self.output_lstm(F.dropout(connector, ratio=self.dropout_ratio, train=train))
        return self.out_word(F.dropout(output_feature, ratio=self.dropout_ratio, train=train))

    def generate(self, context, sentence_limit,gpu=-1):

        sentence = ""
        length = 0
        while length < sentence_limit:
            decoded_feature = self.decode(context, teacher_embed_id=None, train=False)
            if gpu >= 0 :
                word_id = cuda.to_cpu(decoded_feature.data)
            else:
                word_id = decoded_feature.data

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

    model = Image2Seq(output_vocab, feature_num=100, hidden_num=100)

    optimizer = optimizers.Adam()
    # optimizer = optimizers.SGD()
    optimizer.setup(model)
    path = "./data/"
    images = [loader.load(path + data["file_name"]) for data in datas]
    # first run
    features = [model.encode(xp.array([image], dtype=np.float32)) for image in images]
    pickle.dump(features,open("./features.npy","wb"))
    # second run
    #features = pickle.load(open("./features.npy","r"))


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
            context = data["feature"]
            teacher_text = data["wakati_text"]

            # TODO datas[63] and datas[352] has unexpected char code
            try:
                for word in teacher_text:
                    id = model.word2id_output[word]
                    loss = model.decode(context, id, train=True)
                    acc_loss += loss
                    # loss, __context = model.decode(context, id, train=True)
            except:
                # if catch char code error
                # print "data[%d] " % i ,  " has unexpected char code"
                continue

            model.zerograds()  # 勾配をゼロ初期化
            acc_loss.backward()  # 累計損失を使って、誤差逆伝播(誤差の計算)
            acc_loss.unchain_backward()  # truncate # 誤差逆伝播した変数や関数へのreferenceを削除
            optimizer.update()  # 最適化ルーチンの実行
            acc_loss = 0
            model.initialize()

        test_context = datas[epoch % len(datas)]["feature"]
        sentence = model.generate(test_context, 100,args.gpu)
        answer = datas[epoch % len(datas)]["text"]

        print "(%d) epoch " % epoch
        print "*** answer ***\n", answer, "\n"
        print "*** generated *** \n", sentence , "\n"
        print "==================\n\n"

        if epoch % 100 == 0 :
            serializers.save_npz("model_%d.npz" % epoch , model)
            print "model_%d.npz saved" % epoch 


