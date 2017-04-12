from __future__ import print_function, division
from keras.models import Sequential
try:
    from keras.engine.training import slice_X
except:
    from keras.engine.training import _slice_arrays as slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from numpy.random import multinomial
import random
import sys
from collections import defaultdict
from echo_words import CharacterTable, pad, buildCharEncDec

def getPseudowords(text, strategy, pSeg=0.1):
    words = [word for utt in text for word in utt]
    lnCounts = defaultdict(int)
    for wi in words:
        lnCounts[len(wi)] += 1
    mx = max(lnCounts.keys())
    for ii in range(mx):
        if ii not in lnCounts:
            lnCounts[ii] = 0
    total = sum(lnCounts.values())
    lnProbs = [xx[1] / total for xx in
               sorted(lnCounts.items(), key=lambda xx: xx[0])]

    pseuds = []

    if strategy == "geometric":
        #create pseudowords by segmenting utterances geometrically
        for utt in text:
            pseudUtt = []
            boundBefore = 0
            uttChars = "".join(utt)
            for ch in range(len(uttChars)):
                if random.random() < pSeg or ch == len(uttChars) - 1:
                    pseudUtt.append(uttChars[boundBefore:ch + 1])
                    boundBefore = ch + 1

            pseuds.append(pseudUtt)
            #print(utt, pseudUtt)
    elif strategy == "length-matched":
        for utt in text:
            pseudUtt = []
            boundBefore = 0
            if type(utt[0]) == str:
                uttChars = "".join(utt)
            else:
                uttChars = sum(utt, [])
            while boundBefore < len(uttChars):
                nextLn = np.argmax(multinomial(1, lnProbs))
                nextBound = min(nextLn + boundBefore, len(uttChars))
                pseudUtt.append(uttChars[boundBefore:nextBound])
                boundBefore = nextBound

            pseuds.append(pseudUtt)
            #print(utt, pseudUtt)
    else:
        assert(0), "Unknown strategy"

    return pseuds

if __name__ == "__main__":
    path = "br-phono.txt"
    text = [xx.strip().split() for xx in open(path).readlines()]

    chars = ["X"] + list(set("".join(["".join(utt) for utt in text])))
    ctable = CharacterTable(chars)

    outf = file("capacity-stats.tsv", 'w')

    maxlen = 7
    print("max length", maxlen)

    words = [word for utt in text for word in utt]

    pseudWords = getPseudowords(text, strategy="length-matched")
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
    Xreal = np.zeros((len(words), maxlen, len(chars)), dtype=np.bool)
    yreal = np.zeros((len(words), maxlen, len(chars)), dtype=np.bool)
    for ii, word in enumerate(words):
        #print("encoding", word, "at row", ii)
        Xreal[ii] = ctable.encode(pad(word, maxlen, "X"), maxlen)
        yreal[ii] = ctable.encode(pad(word, maxlen, "X"), maxlen)

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(Xreal) - len(Xreal) // 10
    (X_r_train, X_r_val) = (slice_X(Xreal, 0, split_at), 
                            slice_X(Xreal, split_at))
    (y_r_train, y_r_val) = (yreal[:split_at], yreal[split_at:])

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

    for hidden in [10, 20, 40, 80, 160]:
        modReal = buildCharEncDec(hidden, RNN, LAYERS, maxlen, chars,
                                  dropout=0)
        modPseud = buildCharEncDec(hidden, RNN, LAYERS, maxlen, chars,
                                   dropout=0)

        for iteration in range(1, 20):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            modReal.fit(X_r_train, y_r_train, batch_size=BATCH_SIZE, 
                        nb_epoch=1, validation_data=(X_r_val, y_r_val))
            modPseud.fit(X_p_train, y_p_train, batch_size=BATCH_SIZE, 
                         nb_epoch=1, validation_data=(X_p_val, y_p_val))

        (lossReal, accReal) = modReal.evaluate(X_r_val, y_r_val)
        (lossPseud, accPseud) = modPseud.evaluate(X_p_val, y_p_val)

        print(hidden, lossReal, accReal, lossPseud, accPseud,
              sep="\t", file=outf)

    outf.close()
