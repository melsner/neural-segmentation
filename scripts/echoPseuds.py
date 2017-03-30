from __future__ import print_function, division
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from numpy.random import multinomial
import random
import sys
from collections import defaultdict
from echo_words import CharacterTable, pad, buildCharEncDec
from capacityStatistics import getPseudowords
from autoencodeDecodeChars import readText

if __name__ == "__main__":
    path = sys.argv[1]
    out = sys.argv[2]
    text, uttChars, charset = readText(path)
    chars = ["X"] + charset
    ctable = CharacterTable(chars)

    maxlen = 7
    print("max length", maxlen)

    pseudWords = getPseudowords(text, strategy="length-matched", pSeg=0.1)
    pseudWords = [word for utt in pseudWords for word in utt]

    if 1:
        #follows a geometric distribution, which isn't really
        #a good model of word length--- likely to have tons of 1-length
        print("word lengths of pseudovocab")
        lnCounts = defaultdict(int)
        for wi in pseudWords:
            lnCounts[len(wi)] += 1

        tot = sum(lnCounts.values())
        soFar = 0
        for ii in range(1, max(lnCounts.keys()) + 1):
            ct = lnCounts[ii]
            soFar += ct

            print(ii, ct, "\t", soFar / tot)

    print('Vectorization...')
    Xpseud = np.zeros((len(pseudWords), maxlen, len(chars)), dtype=np.bool)
    ypseud = np.zeros((len(pseudWords), maxlen, len(chars)), dtype=np.bool)
    for ii, word in enumerate(pseudWords):
        #print("encoding", word, "at row", ii)
        Xpseud[ii] = ctable.encode(pad(word, maxlen, "X"), maxlen)
        ypseud[ii] = ctable.encode(pad(word, maxlen, "X"), maxlen)

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(Xpseud) - len(Xpseud) // 10
    (X_p_train, X_p_val) = (slice_X(Xpseud, 0, split_at), 
                            slice_X(Xpseud, split_at))
    (y_p_train, y_p_val) = (ypseud[:split_at], ypseud[split_at:])

    RNN = recurrent.LSTM
    BATCH_SIZE = 128
    LAYERS = 1
    hidden = 40

    modPseud = buildCharEncDec(hidden, RNN, LAYERS, maxlen, chars, dropout=.75)

    for iteration in range(1, 20):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        modPseud.fit(X_p_train, y_p_train, batch_size=BATCH_SIZE, 
                     nb_epoch=1, validation_data=(X_p_val, y_p_val))

        toPrint = 10
        preds = modPseud.predict(X_p_train[:toPrint], verbose=0)

        for ii in range(toPrint):
            guess = ctable.decode(preds[ii], calc_argmax=True)
            print(pad(pseudWords[ii], maxlen, "X"))
            print(guess)
            print()

    modPseud.save("%s.h5" % out)
    modPseud.save_weights("%s-weights.h5" % out)

