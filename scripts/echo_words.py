from __future__ import print_function
from keras.models import Sequential
try:
    from keras.engine.training import slice_X
except:
    from keras.engine.training import _slice_arrays as slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Dropout
import numpy as np
import random
import sys
from collections import defaultdict

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = ['X'] + sorted(set(chars) - set(['X']))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        # X = np.zeros((num_rows, len(self.chars)))
        # for i, c in enumerate(C):
        #     X[i, self.char_indices[c]] = 1
        # return X
        X = np.zeros((num_rows, 1), dtype='float32')
        for i, c in enumerate(C):
            X[i, 0] = self.char_indices[c]
        return X

    def decode(self, X):
        # Add small non-zero dimension to index padding characters
        X = np.concatenate([X, np.expand_dims(np.ones_like(X[0]), 0) * 0.01])
        X = X.argmax(-1)
        padding_ix = X.shape[-1] - 1
        return ''.join(self.indices_char[x] if x < padding_ix else 'X' for x in X)

    def dim(self):
        return len(self.chars)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

def buildCharEncDec(hidden, RNN, layers, maxlen, chars, dropout=.3):
    print('Build model...')
    model = Sequential()
    # "Encode" the input sequence using an RNN,
    # producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).


    # model.add(RNN(hidden, input_shape=(maxlen, len(chars)), 
    #               name="encoder-rnn"))

    model.add(Dropout(dropout, input_shape=(maxlen, len(chars)),
                       noise_shape=(1, maxlen, 1)))
    model.add(RNN(hidden, name="encoder-rnn"))

    # As the decoder RNN's input, repeatedly provide with the 
    # last hidden state of
    # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    model.add(RepeatVector(maxlen, name="encoding"))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for ii in range(layers):
        # By setting return_sequences to True, return not only the last output 
        # but all the outputs so far in the form of (nb_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below
        # expects the first dimension to be the timesteps.
        model.add(RNN(hidden, return_sequences=True, 
                      name="decoder%i" % ii))

    # Apply a dense layer to the every temporal slice of an input.
    # For each step
    # of the output sequence, decide which character should be chosen.
    model.add(TimeDistributed(Dense(len(chars), name="dense"), name="td"))
    model.add(Activation('softmax', name="softmax"))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    return model

def pad(word, maxlen, padChar):
    clipped = word[:maxlen]
    if type(word) == str:
        return clipped + (padChar * (maxlen - len(clipped)))
    else:
        return clipped + [padChar,] * (maxlen - len(clipped))