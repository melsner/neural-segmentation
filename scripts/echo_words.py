from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
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
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        X = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

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

if __name__ == "__main__":
    path = "br-phono.txt"
    text = open(path).read()
    words = text.split()
    print('corpus length:', len(words))

    chars = ["X"] + list(set("".join(words)))
    print('total chars:', len(chars))
    ctable = CharacterTable(chars)

    maxlen = 7
    print("max length", maxlen)

    print('Vectorization...')
    X = np.zeros((len(words), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(words), maxlen, len(chars)), dtype=np.bool)
    for ii, word in enumerate(words):
        #print("encoding", word, "at row", ii)
        X[ii] = ctable.encode(pad(word, maxlen, "X"), maxlen)
        y[ii] = ctable.encode(pad(word, maxlen, "X"), maxlen)

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(X) - len(X) // 10
    (X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
    (y_train, y_val) = (y[:split_at], y[split_at:])

    RNN = recurrent.LSTM
    #the memory capacity depends on this parameter
    #at 128, 7-letter words can be reproduced, but at 10 they cannot
    #20 gets to 83% at 5, 93 at 14, 97 at 23, 98 by 40
    #30 is in the 97 range at 14 epochs, 90% at 5
    #40 gets >99 acc in about 14 epochs
    #80 gets to 96 in only 4 epochs and 99 in 7
    HIDDEN_SIZE = 40 #128
    BATCH_SIZE = 128
    LAYERS = 2

    model = buildCharEncDec(hidden, RNN, LAYERS, maxlen, chars)

    # Train the model each generation and show predictions against
    # the validation dataset.
    for iteration in range(1, 15):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
                  validation_data=(X_val, y_val))
        # Select 10 samples from the validation set at random so we 
        # can visualize errors.
        for i in range(10):
            ind = np.random.randint(0, len(X_val))
            rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowX, verbose=0)
            q = ctable.decode(rowX[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print('Q', q)
            print('T', correct)
            if correct == guess:
                print(colors.ok + 'Y' + colors.close, end=" ")
            else:
                print(colors.fail + 'N' + colors.close, end=" ")
            print(guess)
            print('---')

    model.save("word-echo.h5")
    model.save_weights("word-echo-weights.h5")
