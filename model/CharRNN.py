import numpy as np
from chainer import Variable, Chain
import chainer.functions as F
import chainer.links as L


class CharRNN(Chain):
    def __init__(self, n_vocab, n_units):
        super(CharRNN, self).__init__(
                embed=F.EmbedID(n_vocab, n_units),
                l1=L.LSTM(n_units, n_units),
                l2=L.LSTM(n_units, n_units),
                l3=L.Linear(n_units, n_vocab),
        )


    def forward_one_step(self, x_data, y_data, state, train=True, dropout_ratio=0.5):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)
        """
         c : context
         h : output
        """

        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, ratio=dropout_ratio, train=train))
        c1 = self.l1.c
        h2 = self.l2(F.dropout(h1, ratio=dropout_ratio, train=train))
        c2 = self.l2.c
        y = self.l3(F.dropout(h2, ratio=dropout_ratio, train=train))
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

        if train:
            return state, F.softmax_cross_entropy(y, t)
        else:
            return state, F.softmax(y)


def make_initial_state(n_units, batchsize=50, train=True):
    return {name: Variable(np.zeros((batchsize, n_units), dtype=np.float32),
                           volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}

