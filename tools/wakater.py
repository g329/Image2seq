#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
import numpy as np
import sys
import codecs
import pickle
from tools.feature_extractor import FeatureExtractor
from tools.io_util import ImageLoader

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

import MeCab


def make_vocab(text_array, is_input=False):
    """
    make vocab for encoder-decoder model

    :param text_array: ["私は生徒です","私は漁師です"]
    :return: vocabs ["私","は","生徒","漁師","です","<end>"] if output == True , add <start>
    """
    m = MeCab.Tagger()
    vocabs = set()
    for text in text_array:
        n = m.parseToNode(text.encode('utf-8', 'ignore'))
        n = n.next
        while n:
            word = n.surface.decode('utf-8', 'ignore')
            if word == "":
                n = n.next
                continue
            vocabs.add(word)
            n = n.next
    if is_input == True:
        # in image2seq , dont use <start>
        vocabs.add("<start>")

    vocabs.add("<end>")

    return vocabs

def wakati(text_origin,train=False):
    """

    :param text: "私は生徒です"
    :return:  ["私","は","生徒","です"]
    """
    m = MeCab.Tagger()
    watati_text = []

    n = m.parseToNode(text_origin.encode('utf-8', 'ignore'))
    n = n.next
    while n:
        word = n.surface.decode('utf-8', 'ignore')
        if word == "":
            n = n.next
            continue
        watati_text.append(word)
        n = n.next
    watati_text.append("<end>")

    return  watati_text


if __name__ == "__main__":
    texts = ["私は生徒です", "私は漁師です"]
    vocab = make_vocab(texts, is_input=False)
    wakati = wakati(texts[0])
    for text in wakati:
        for t in text:
            print t
