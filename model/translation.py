#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.optimizers
import math
import numpy as np
from chainer import Variable, Chain
import chainer.functions as F
import chainer.links as L


class Translation(Chain):
    def __init__(self, src_vocab_size, src_embed_size, trg_vocab_size, trg_embed_size, n_units):
        super(Translation, self).__init__(
                # 入力を圧縮するための層
                src_embed=F.EmbedID(src_vocab_size, src_embed_size),

                # encode layer
                src_l1=L.LSTM(src_embed_size, n_units),

                # encode -> decode
                pq=L.LSTM(n_units, n_units),

                # decode layer
                trg_l1=L.LSTM(n_units, trg_embed_size),
                trg_l2=L.LSTM(trg_embed_size, trg_vocab_size),

                # input layer for test
                trg_embed=F.EmbedID(trg_vocab_size, n_units),
        )

        self.HIDDEN_SIZE = n_units
        self.dict = {}

    def __call__(self, src_sentence, trg_sentence, training):
        """

        :param src_sentence: Japanese ["彼", "は","速く" ,"走る"]
        :param trg_sentence: English [ "He", "runs", "fast"]
        :param training: True , False
        :return:
        """

        src_sentence = [self.dict[word] for word in src_sentence]
        trg_sentence = [self.dict[word] for word in src_sentence]

        c = Variable(np.zeros((1, self.HIDDEN_SIZE), dtype=np.float32))

        for word in reversed(src_sentence):
            x = Variable(np.array([[word]], dtype=np.int32))
            embed = F.tanh(self.src_embed)
            p = self.src_l1(embed)
            context = self.src_l1.c

        # encorder -> decorder
        self.pq.c = context
        q = self.pq(p)

        # 学習時：y : 正解の翻訳
        if training:
            acc_loss = np.zeros((), dtype=np.float32)

            for word in trg_sentence:
                j = F.tanh(self.trg_l1(q))
                y = self.trg_l2
                t = Variable(np.array([[word]], dtype=np.int32))
                acc_loss += F.softmax_cross_entropy(y, t)
                # copy context
                self.trg_embed.c = self.trg_l1.c
                q = self.trg_embed(t)

            return acc_loss
        else:
            hyp_sentence = []
            # 単語の生成制限
            while len(hyp_sentence) < 100:
                j = F.tanh()
