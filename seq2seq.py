#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
import numpy as np
import sys
import codecs

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)


class Seq2Seq(chainer.Chain):
    dropout_ratio = 0.5

    def __init__(self, input_vocab, output_vocab, feature_num, hidden_num):
        """

        :param input_vocab: array of input space vocab
        :param output_vocab: array of output space vocab
        :param feature_num: size of feature layer
        :param hidden_num: size of hidden layer
        :return:
        """
        dict = {}
        self.id2word_input = {}
        self.word2id_input = {}
        for id, word in enumerate(input_vocab):
            self.id2word_input[id] = word
            self.word2id_input[word] = id

        self.id2word_output = {}
        self.word2id_output = {}
        for id, word in enumerate(output_vocab):
            self.id2word_output[id] = word
            self.word2id_output[word] = id

        self.input_vocab_size = len(self.word2id_input)
        self.output_vocab_size = len(self.word2id_output)

        super(Seq2Seq, self).__init__(
                # encoder
                word_vec=L.EmbedID(self.input_vocab_size, feature_num),
                # input_vec=L.Linear(feature_num, hidden_num),
                input_vec=L.LSTM(feature_num, hidden_num),

                # connect layer
                context_lstm=L.LSTM(hidden_num, hidden_num),

                # decoder
                output_lstm=L.LSTM(hidden_num, feature_num),
                out_word=L.Linear(feature_num, self.output_vocab_size),
        )

    def encode(self, src_text, train):
        """

        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :return: context vector
        """
        for word in src_text:
            word = chainer.Variable(np.array([[word]], dtype=np.int32))
            embed_vector = F.tanh(self.word_vec(word))
            input_feature = self.input_vec(embed_vector)
            context = self.context_lstm(F.dropout(input_feature, ratio=self.dropout_ratio, train=train))

        return context

    def decode(self, context, teacher_embed_id, train):
        """

        :param context: context vector which maked `encode` function
        :return: decoded embed vector
        """

        # output_feature = self.output_lstm(F.dropout(context, ratio=self.dropout_ratio, train=train))
        # output_feature = F.dropout(self.output_lstm(context), ratio=self.dropout_ratio, train=train)
        output_feature = self.output_lstm(context)
        # predict_embed_id = self.out_word(F.dropout(output_feature, ratio=self.dropout_ratio, train=train))
        # predict_embed_id = F.dropout(self.out_word(output_feature), ratio=self.dropout_ratio, train=train)
        predict_embed_id = F.tanh(self.out_word(output_feature))
        if train:
            t = np.zeros(1, dtype=np.int32)
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
        self.input_vec.reset_state()
        self.context_lstm.reset_state()
        self.output_lstm.reset_state()
        # self.lstm.reset_state()
        # h = self.image_vec(F.dropout(image_feature, ratio=self.dropout_ratio, train=train))
        # self.lstm(F.dropout(h, ratio=self.dropout_ratio, train=train))

    def generate(self, start_word_id, sentence_limit):

        context = self.encode([start_word_id], train=False)

        sentence = ""
        length = 0
        while length < sentence_limit:
            word_id, context = self.decode(context, teacher_embed_id=None, train=False)
            word = self.id2word_output[np.argmax(word_id.data)]
            if word == "<end>":
                break
            sentence = sentence + word + " "
            length += 1
        return sentence


if __name__ == "__main__":
    # input_vocab = ["<start>", "I", "am", "a", "student", "<end>"]
    # output_vocab = [u"私は", u"生徒", u"です","<end>"]

    input_vocab = ["<start>", u"黄昏に", u"天使の声", u"響く時，", u"聖なる泉の前にて", u"待つ", "<end>"]
    output_vocab = [u"5時に", u"噴水の前で", u"待ってます", "<end>"]

    model = Seq2Seq(input_vocab, output_vocab, feature_num=4, hidden_num=10)

    # optimizer = optimizers.Adam()
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    # print "input : ".join(input_vocab[1:6])


    for _ in range(20000):
        input = []
        model.initialize()
        for word in reversed(input_vocab):
            input.append(model.word2id_input[word])

        context = model.encode(input, train=True)
        acc_loss = 0
        for word in output_vocab:
            id = model.word2id_output[word]
            loss = model.decode(context, id, train=True)
            acc_loss += loss

        model.zerograds()  # 勾配をゼロ初期化
        acc_loss.backward()  # 累計損失を使って、誤差逆伝播(誤差の計算)
        acc_loss.unchain_backward()  # truncate # 誤差逆伝播した変数や関数へのreferenceを削除
        acc_loss = 0
        optimizer.update()  # 最適化ルーチンの実行

        model.initialize()
        start = model.word2id_input["<start>"]
        sentence = model.generate(start, 7)

        print "input : ", "".join(input_vocab[1:6])
        print "-> ", sentence
        print
