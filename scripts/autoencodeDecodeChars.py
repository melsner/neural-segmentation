from __future__ import print_function, division
import argparse
from keras.models import Model, Sequential, load_model
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Input, Reshape, Merge, merge, Lambda, Dropout
from keras import backend as K
import tensorflow as tf
import numpy as np
import cPickle as pickle
import random
import sys
import os
import re
import copy
from collections import defaultdict
from echo_words import CharacterTable, pad
from capacityStatistics import getPseudowords
from score import *

def uttsToCharVectors(text, maxchar, ctable):
    nUtts = len(text)
    X = np.zeros((nUtts, maxchar, ctable.dim()), dtype=np.bool)

    for ui, utt in enumerate(text):
        X[ui] = ctable.encode(pad(utt[:maxchar], maxchar, "X"), maxchar)

    return X

def uttsToWordVectors(text, maxutts, maxlen, chars):
    nUtts = len(text)
    X = np.zeros((nUtts, maxutt, maxlen, ctable.dim()), dtype=np.bool)

    for ui, utt in enumerate(text):
        corr = max(0, maxutt - len(utt))

        for ii in range(corr):
            X[ui, ii, :] = ctable.encode("X" * maxlen, maxlen)

        for ii, word in enumerate(utt[:maxutt]):
            X[ui, corr + ii, :] = ctable.encode(pad(word, maxlen, "X"), maxlen)

    return X

def inputSliceClosure(dim):
    return lambda xx: xx[:, dim, :, :]

def outputSliceClosure(dim):
    return lambda xx: xx[:, dim, :]

def batchIndices(Xt, batchSize):
    xN = Xt.shape[0]
    if xN % batchSize > 0:
        lastBit = [(xN - xN % batchSize, xN)]
    else:
        lastBit = []

    return [slice(aa, bb) for (aa, bb) in 
            zip(range(0, xN, batchSize),
                range(batchSize, xN, batchSize)) + lastBit]

def segsToX(textChars, segs, maxutt, maxlen, ctable):
    nUtts = len(textChars)
    X = np.zeros((nUtts, maxutt, maxlen, ctable.dim()), dtype=np.bool)
    deletedChars = np.zeros((nUtts,))
    oneWord = np.zeros((nUtts,))

    for ui,uttChars in enumerate(textChars):
        uttWds = np.where(segs[ui])[0]
        #print("uttwds", uttWds)
        corr = max(0, maxutt - len(uttWds))

        for ii in range(corr):
            X[ui, ii, :] = ctable.encode("X" * maxlen, maxlen)

        prev = 0
        for ii,bd in enumerate(uttWds[:maxutt]):
            word = uttChars[prev:bd + 1]
            assert(word != "")
            deletedChars[ui] += max(0, len(word) - maxlen)
            if len(word) == 1:
                oneWord[ui] += 1
            #print("deleted", max(0, len(word) - maxlen), "chars in", word)

            # print(uttChars, "segment at", prev, bd, word)
            X[ui, corr + ii, :] = ctable.encode(pad(word, maxlen, "X"), maxlen)
            prev = bd + 1

        #print("checking deleted words", uttWds[maxutt:])
        for ii,bd in enumerate(uttWds[maxutt:]):
            word = uttChars[prev:bd + 1]
            assert(word != "")
            #print("deleted", len(word), "chars in entire word", word)
            deletedChars[ui] += len(word)
            prev = bd + 1

    return X, deletedChars, oneWord

def sampleSegs(utts, pSegs):
    smp = np.random.uniform(size=pSegs.shape)
    maxchar = pSegs.shape[1]
    indics = smp < pSegs
    for ui,utt in enumerate(utts):
        if len(utt) < maxchar:
            indics[ui, len(utt) - 1] = 1
            indics[ui, len(utt):] = 0
        else:
            indics[ui, maxchar - 1] = 1

    return indics

def lossByUtt(model, Xb, yb, BATCH_SIZE, metric="logprob"):
    preds = model.predict(Xb, batch_size=BATCH_SIZE, verbose=0)

    if metric == "logprob":
        logP = np.log(preds)
        pRight = logP * yb
        #sum out word, char, len(chars)
        return pRight.sum(axis=(1, 2, 3))
    else:
        right = (np.argmax(preds, axis=-1) == np.argmax(yb, axis=-1))
        rightPerUtt = right.sum(axis=(1,2))

        # print("debug lossbyutt")
        # print("pred", np.argmax(preds, axis=-1)[0])
        # print("y", np.argmax(yb, axis=-1)[0])
        # print("corr", right[0])
        # print("shape of right", right.shape)
        # print("sum of right", rightPerUtt[0])

        return rightPerUtt

def guessSegTargets(scores, segs, priorSeg, metric="logprob"):
    #sample x utterance
    scores = np.array(scores)

    if metric == "logprob":
        MM = np.max(scores, axis=0, keepdims=True)
        eScores = np.exp(scores - MM)
        #approximately the probability of the sample given the data
        pSeg = eScores / eScores.sum(axis=0, keepdims=True)
    else:
        pSeg = scores / scores.sum(axis=0, keepdims=True)

    #sample x utterance x seg 
    segmat = np.array(segs)

    #proposal prob
    priorSeg = np.expand_dims(priorSeg, 0)
    qSeg = segmat * priorSeg + (1 - segmat) * (1 - priorSeg)

    wts = np.expand_dims(pSeg, -1) / qSeg
    wts = wts / wts.sum(axis=0, keepdims=True)

    #print("score distr:", dist[:10])

    # print("shape of segment matrix", segmat.shape)
    # print("losses for utt 0", scores[:, 0])
    # print("top row of distr", pSeg[:, 0])
    # print("top row of correction", qSeg[:, 0])
    # print("top row of weights", wts[:, 0])

    #sample x utterance x segment
    nSamples = segmat.shape[0]
    wtSegs = segmat * wts

    # for si in range(nSamples):
    #     print("seg vector", si, segmat[si, 0, :])
    #     print("est posterior", pSeg[si, 0])
    #     print("q", qSeg[si, 0])
    #     print("weight", wts[si, 0])
    #     print("contrib", wtSegs[si, 0])

    segWts = wtSegs.sum(axis=0)
    best = segWts > .5

    # print("top row of wt segs", segWts[0])
    # print("max segs", best[0])
    return segWts, best

def sampleSegments(model, text, maxutt, maxlen, ctable, verbose=False):
    assert(0)
"""    nUtts = len(text)
    X = np.zeros((nUtts, maxutt, maxlen, ctable.dim()), dtype=np.bool)

    pseuds = getPseudowords(text, "geometric", pSeg=.3)

    for ui, utt in enumerate(pseuds):
        corr = max(0, maxutt - len(utt))

        for ii in range(corr):
            X[ui, ii, :] = ctable.encode("X" * maxlen, maxlen)

        for ii, word in enumerate(utt[:maxutt]):
            X[ui, corr + ii, :] = ctable.encode(pad(word, maxlen, "X"), maxlen)

    y = X[:, ::-1, :]

    preds = model.predict(X, verbose=0)
    preds = preds[:, ::-1, :]

    if verbose:
        for utt in range(10):
            print(" ".join(["X" * maxlen for ii in
                            range(maxutt - len(pseuds[utt]))] + 
                           [pad(word, maxlen, "X") for word in pseuds[utt]]))
            for wi in range(maxutt):
                guess = ctable.decode(preds[utt, wi], calc_argmax=True)
                print(guess, end=" ")
            print()
            # for wi in range(maxutt):
            #     print(np.argmax(y[utt, wi], axis=-1), end=" ")
            # print()
            # for wi in range(maxutt):
            #     print(np.argmax(preds[utt, wi], axis=-1), end=" ")
            # print()

            # for wi in range(maxutt):
            #     charsRight = (np.argmax(preds[utt, wi], axis=-1) == 
            #                   np.argmax(y[utt, wi], axis=-1))
            #     print(charsRight, charsRight.sum(), end=" ")
            # print()

            # rowChars = (np.argmax(preds[utt], axis=-1) ==
            #             np.argmax(y[utt], axis=-1))
            # print("whole-row argmax", rowChars.shape, rowChars.sum())

            print("\n")

    allChars = (np.argmax(preds, axis=-1) == np.argmax(y, axis=-1))
    # print("whole-set argmax", allChars.shape, allChars.sum())

    return (allChars.sum() / allChars.size)"""

def reconstruct(chars, segs, maxutt, wholeSent=False):
    uttWds = np.where(segs)[0][:maxutt]
    prev = 0
    words = []
    for ii,bd in enumerate(uttWds):
        word = chars[prev:bd + 1]
        words.append(word)
        assert(word != "")
        prev = bd + 1

    if wholeSent:
        if prev < len(chars):
            word = chars[prev:len(chars)]
            words.append(word)

    return words

def matToSegs(segmat, text):
    if type(text[0][0]) == str:
        text = ["".join(utt) for utt in text]    
    else:
        text = [sum(utt, []) for utt in text]

    res = []
    for utt in range(len(text)):
        thisSeg = segmat[utt]
        #pass dummy max utt length to reconstruct everything
        rText = reconstruct(text[utt], thisSeg, 100, wholeSent=True)
        res.append(rText)

    return res

def writeSolutions(logdir, model, segmenter, allBestSeg, text, iteration):
    model.save(logdir + "/model-%d.h5" % iteration)
    segmenter.save(logdir + "/segmenter-%d.h5" % iteration)

    segmented = matToSegs(allBestSeg, text)

    logfile = file(logdir + "/segmented-%d.txt" % iteration, 'w')
    if type(text[0][0]) == str:
        for line in segmented:
            print(" ".join(line), file=logfile)
    else:
        for line in segmented:
            print(" || ".join([" ".join(wi) for wi in line]), "||",
                  file=logfile)
    logfile.close()

def printSegScore(text, allBestSeg):
    segmented = matToSegs(allBestSeg, text)

    #print(segmented)
    #print(text)

    (bp,br,bf) = scoreBreaks(text, segmented)
    (swp,swr,swf) = scoreWords(text, segmented)
    (lp,lr,lf) = scoreLexicon(text, segmented)
    print("SP %4.2f SR %4.2f SF %4.2f" % (100 * swp, 100 * swr, 100 * swf))
    print("BP %4.2f BR %4.2f BF %4.2f" % (100 * bp, 100 * br, 100 * bf))
    print("LP %4.2f LR %4.2f LF %4.2f" % (100 * lp, 100 * lr, 100 * lf))

def writeLog(iteration, epochLoss, epochDel, text, allBestSeg, logfile):
    segmented = matToSegs(allBestSeg, text)

    (bp,br,bf) = scoreBreaks(text, segmented)
    (swp,swr,swf) = scoreWords(text, segmented)
    (lp,lr,lf) = scoreLexicon(text, segmented)

    print("\t".join(["%g" % xx for xx in [
                    iteration, epochLoss, epochDel, bp, br, bf, swp, swr, swf,
                    lp, lr, lf]]), file=logfile)

def realize(rText, maxlen, maxutt):
    items = (["X" * maxlen for ii in range(maxutt - len(rText))] + 
             [pad(word, maxlen, "X") for word in rText])
    def delist(wd):
        if type(wd) == list:
            return "".join(wd)
        else:
            return wd
    items = [delist(wd) for wd in items]

    return " ".join(items)

def readText(path):
    lines = open(path).readlines()
    if any(("||" in li for li in lines)):
        reader = "arpa"
    else:
        reader = "brent"

    print("Guessing reader setting:", reader)

    if reader == "arpa":
        text = []
        chars = []
        for line in lines:
            #last element of list is []
            words = line.strip().split("||")[:-1]
            words = [wi.split() for wi in words]
            text.append(words)
            chars.append(sum(words, []))

        charset = set()
        for li in chars:
            charset.update(li)
        charset = list(charset)

    else:
        text = [xx.strip().split() for xx in lines]
        chars = ["".join(utt) for utt in text]
        charset = list(set("".join(chars)))

    return text, chars, charset

def readMFCCs(path, filter_file=None):
    basename = re.compile('.*/(.+)\.mfcc')
    filelist = sorted([path+x for x in os.listdir(path) if x.endswith('.mfcc')])
    idlist = [basename.match(x).group(1) for x in filelist]

    if filter_file:
        to_keep = {}
        with open(args.splitfile, 'rb') as s:
            for line in s:
                if line.strip() != None:
                    name, start, end = line.strip().split()
                    if name in to_keep:
                        to_keep[name].append((int(math.floor(float(start)*100)),int(math.ceil(float(end)*100))))
                    else:
                        to_keep[name] = [(int(math.floor(float(start)*100)),int(math.ceil(float(end)*100)))]

    mfcc_lists = {}

    for p in filelist:
        mfcc_counter = 0
        file_id = basename.match(p).group(1)
        mfcc_lists[file_id] = []
        with open(p, 'rb') as f:
            for line in f:
                if line.strip() != '[':
                    if line.strip().startswith('['):
                        line = line.strip()[1:]
                    if line.strip().endswith(']'):
                        line = line.strip()[:-1]
                    else:
                        mfcc = map(float, line.strip().split())
                    mfcc_lists[file_id].append(mfcc)
        mfcc_lists[file_id] = np.asarray(mfcc_lists[file_id])
    return mfcc_lists

def splitMFCCs(mfccs,intervals,segs):
    # Filter out non-speech portions
    mfcc_intervals = {}
    for doc in segs:
        mfcc_intervals[doc] = np.zeros((0,39))
        for i in intervals[doc]:
            sf, ef = int(np.rint(float(i[0]*100))), int(np.rint(float(i[1]*100)))
            mfcc_intervals[doc] = np.append(mfcc_intervals[doc], mfccs[doc][sf:ef,:], 0)

    # Split mfcc intervals according to segs
    out = {}
    for doc in mfcc_intervals:
        out[doc] = np.split(mfcc_intervals[doc], np.asarray(range(mfcc_intervals[doc].shape[0]))[np.where(segs[doc])])[1:]
    return out


def timeSeg2frameSeg(timeseg_file):
    intervals = {}
    speech = {}
    offsets = {}
    seg = 0
    with open(timeseg_file, 'rb') as f:
        for line in f:
            if line.strip() != '':
                doc, start, end = line.strip().split()
                if doc in intervals:
                    if float(start) == intervals[doc][-1][1]:
                        intervals[doc][-1] = (intervals[doc][-1][0],float(end))
                    else:
                        intervals[doc].append((float(start),float(end)))
                else:
                    intervals[doc] = [(float(start),float(end))]
                s, e = int(np.rint(float(start)*100)), int(np.rint(float(end)*100))
                if doc in speech:
                    last = speech[doc][-1][1] + offsets.get(doc, 0)
                else:
                    last = 0
                if last < s:
                    seg = 1
                    if doc in offsets:
                        offsets[doc] += s - last
                    else:
                        offsets[doc] = s - last
                else:
                    seg = 0
                offset = offsets.get(doc, 0)
                if doc in speech:
                    speech[doc].append((s-offset,e-offset,seg))
                else:
                    speech[doc] = [(s-offset,e-offset,seg)]

    segs = {}
    for doc in speech:
        segs[doc] = np.zeros((speech[doc][-1][1]))
        for seg in speech[doc]:
            segs[doc][seg[0]] = 1.0

    return intervals, segs 

def frameSeg2timeSeg(intervals, seg_f):
    for doc in intervals:
        offset = last_interval = last_seg = 0
        this_frame = 0
        next_frame = 1
        seg_t = []
        for i in intervals[doc]:
            # Interval boundaries in seconds (time)
            st, et = i
            # Interval boundaries in frames
            sf, ef = int(np.rint(float(st)*100)), int(np.rint(float(et)*100))

            offset += sf - last_interval
            last_interval = ef
            while this_frame + offset < ef:
                if next_frame >= seg_f[doc].shape[0] or np.allclose(seg_f[doc][next_frame], 1):
                    if last_seg+offset == sf:
                        start = st
                    else:
                        start = float(last_seg+offset)/100
                    if next_frame+offset == ef:
                        end = et
                    else:
                        end = float(next_frame+offset)/100
                    seg_t.append((start,end))
                    last_seg = next_frame
                this_frame += 1
                next_frame += 1

    return seg_t



def reconstructFullMFCCs(speech_in, nonspeech_in):
    speech = copy.deepcopy(speech_in)
    nonspeech = copy.deepcopy(nonspeech_in)
    out = []
    offset = 0
    last = 0
    s = speech.pop(0)
    ns = nonspeech.pop(0)
    while ns[1] != None:
        if last == ns[0]:
            out.append((ns[0],ns[1],0))
            offset += ns[1] - ns[0]
            last = ns[1]
            if len(nonspeech) > 0:
                ns = nonspeech.pop(0)
            else:
                ns = (np.inf, None)
        else:
            out.append((s[0] + offset, s[1] + offset, s[2]))
            last = out[-1][1]
            if len(speech) > 0:
                s = speech.pop(0)
            else:
                s = (np.inf, None)
    if s[1] != None:
        out.append((s[0]+offset, s[1]+offset, s[2]))
    
    for x in speech:
        out.append((x[0]+offset, x[1]+offset, x[2]))

    for x in out:
        print('%s %s %s' %x)
    return out
            

if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    #parser.add_argument("pseudWeights")
    parser.add_argument("--uttHidden", default=400)
    parser.add_argument("--wordHidden", default=40)
    parser.add_argument("--segHidden", default=100)
    parser.add_argument("--wordDropout", default=.5)
    parser.add_argument("--charDropout", default=.5)
    parser.add_argument("--logfile", default=None)
    parser.add_argument("--acoustic", action='store_true')
    parser.add_argument("--segfile", action=None)
    args = parser.parse_args()

    path = args.data
    #pseudWeights = args.pseudWeights

    # path = sys.argv[1]
    # pseudWeights = sys.argv[2] #"pseud-echo-weights.h5"

    if args.acoustic:
        mfccs_full = readMFCCs(path)
        intervals, segs_frames = timeSeg2frameSeg(args.segfile)
        words_init_frames = splitMFCCs(mfccs_full, intervals, segs_frames)
    else:
        text, uttChars, charset = readText(path)

    exit()

    print('corpus length:', len(text))
    chars = ["X"] + charset
    print('total chars:', len(chars))
    ctable = CharacterTable(chars)

    if args.logfile == None:
        logdir = "logs/" + str(os.getpid())
    else:
        logdir = "logs/" + args.logfile
    os.mkdir(logdir)
    print("Logging at", logdir)
    logfile = file(logdir + "/log.txt", "w")
    print("\t".join([
                    "iteration", "epochLoss", "epochDel", 
                    "bp", "br", "bf", "swp", "swr", "swf",
                    "lp", "lr", "lf"]), file=logfile)

    maxlen = 7
    maxutt = 10
    maxchar = 30
    hidden = int(args.wordHidden) #40
    wordDecLayers = 1
    uttHidden = int(args.uttHidden) #400
    segHidden = int(args.segHidden) #100
    wordDropout = float(args.wordDropout) #.5
    charDropout = float(args.charDropout) #.5
    RNN = recurrent.LSTM
    reverseUtt = True
    BATCH_SIZE = 128
    N_SAMPLES = 50
    DEL_WT = 50
    ONE_LETTER_WT = 10
    METRIC = "logprob"

    nUtts = len(text)

    if args.acoustic:
        pass 
    else:
        XC = uttsToCharVectors(uttChars, maxchar, ctable)

    wordEncoder = Sequential()
    wordEncoder.add(Dropout(charDropout, input_shape=(maxlen, len(chars)),
                       noise_shape=(1, maxlen, 1)))
    wordEncoder.add(RNN(hidden, name="encoder-rnn"))
    # wordEncoder.add(RNN(hidden, input_shape=(maxlen, len(chars)),
    #                     name="encoder-rnn"))

    wordDecoder = Sequential()
    wordDecoder.add(RepeatVector(input_shape=(hidden,),
                                 n=maxlen, name="encoding"))
    for ii in range(wordDecLayers):
        wordDecoder.add(RNN(hidden, return_sequences=True, 
                            name="decoder%i" % ii))

    wordDecoder.add(TimeDistributed(Dense(len(chars), name="dense"), name="td"))
    wordDecoder.add(Activation('softmax', name="softmax"))

    wordEncoder.load_weights(pseudWeights, by_name=True)
    wordDecoder.load_weights(pseudWeights, by_name=True)

    print("Build full model...")

    #input word encoders
    inp = Input(shape=(maxutt, maxlen, len(chars)))
    # encs = []
    # for ii in range(maxutt):
    #     slicer = Lambda(inputSliceClosure(ii),
    #                     output_shape=(maxlen, len(chars)))(inp)

    #     encs.append(wordEncoder(slicer))

    ## stack word encodings
    # mer = merge(encs, mode="concat")
    # resh = Reshape(target_shape=(maxutt, hidden))(mer)
    resh = TimeDistributed(wordEncoder)(inp)

    resh = TimeDistributed(Dropout(wordDropout, noise_shape=(1,)))(resh)

    #utterance encode-decode
    encoderLSTM = RNN(uttHidden, name="utt-encoder")(resh)
    repeater = RepeatVector(maxutt, input_shape=(hidden,))(encoderLSTM)
    decoder = RNN(uttHidden, return_sequences=True, 
                  name="utt-decoder")(repeater)
    decOut = TimeDistributed(Dense(hidden, activation="linear"))(decoder)

    #output word decoders
    # decs = []
    # for ii in range(maxutt):
    #     slicer = Lambda(outputSliceClosure(ii),
    #                     output_shape=(hidden,))(decOut)
    #     decs.append(wordDecoder(slicer))

    ## stack word decodings
    # mer2 = merge(decs, mode="concat", concat_axis=1)
    # resh2 = Reshape(target_shape=(maxutt, maxlen, len(chars)))(mer2)
    resh2 = TimeDistributed(wordDecoder)(decOut)

    model = Model(input=inp, output=resh2)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    segmenter = Sequential()
    segmenter.add(RNN(segHidden, input_shape=(maxchar, len(chars)),
                      return_sequences=True, name="segmenter"))
    segmenter.add(TimeDistributed(Dense(1)))
    segmenter.add(Activation("sigmoid"))
    segmenter.compile(loss="binary_crossentropy",
                      optimizer="adam")
    segmenter.summary()

    print("Training setup...")

    ## pretrain
    for iteration in range(10):
        print()
        print('-' * 50)
        print('Iteration', iteration)

        ## set up matrices for pretraining
        utts = uttChars
        pSegs = .2 * np.ones((len(utts), maxchar))

        segs = sampleSegs(utts, pSegs)

        X_train,deletedChars,oneLetter = segsToX(utts, segs,
                                                 maxutt, maxlen, ctable)
        if reverseUtt:
            y_train = X_train[:, ::-1, :]
        else:
            y_train = X_train

        print("Actual deleted chars:", deletedChars.sum())

        model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1)

        toPrint = 10
        preds = model.predict(X_train[:toPrint], verbose=0)
        if reverseUtt:
            preds = preds[:, ::-1, :]

        for utt in range(toPrint):
            thisSeg = segs[utt]
            rText = reconstruct(utts[utt], thisSeg, maxutt)

            print(realize(rText, maxlen, maxutt))

            for wi in range(maxutt):
                guess = ctable.decode(preds[utt, wi], calc_argmax=True)
                print(guess, end=" ")
            print("\n")

    allBestSegs = np.zeros((X_train.shape[0], maxchar))

    for iteration in range(81):
        print()
        print('-' * 50)
        print('Iteration', iteration)

        ## mixed on/off policy learning?
        # alpha = min(1, (iteration / 30.))

        epochLoss = 0
        epochDel = 0
        epochOneL = 0

        for batch, inds in enumerate(batchIndices(X_train, BATCH_SIZE)):
            printSome = False
            if batch % 25 == 0:
                print("Batch:", batch)
                if batch == 0:
                    printSome = True

            XCb = XC[inds]
            utts = uttChars[inds]

            if iteration < 10:
                nSamples = N_SAMPLES
                pSegs = .1 * np.ones((len(utts), maxchar))
            else:
                nSamples = N_SAMPLES
                pSegs = segmenter.predict(XCb, verbose=0)
                #original shape has trailing 1
                pSegs = np.squeeze(pSegs, -1)

                #smooth this out a bit?
                pSegs = .9 * pSegs + .1 * .5 * np.ones((len(utts), maxchar))
                #print("pseg shape", pSegs.shape)

            ## interpolate policies
            # pSegsOff = .05 * np.ones((len(utts), maxchar))
            # pSegsOn = segmenter.predict(XCb, verbose=0)
            # pSegsOn = np.squeeze(pSegsOn, -1)
            # pSegs = (1 - alpha) * pSegsOff + alpha * pSegsOn

            scores = []
            segSamples = []
            dels = []
            for sample in range(nSamples):
                segs = sampleSegs(utts, pSegs)

                Xb,deletedChars,oneLetter = segsToX(utts, segs,
                                                    maxutt, maxlen, ctable)
                if reverseUtt:
                    yb = Xb[:, ::-1, :]
                else:
                    yb = Xb

                loss = lossByUtt(model, Xb, yb, BATCH_SIZE, metric=METRIC)
                scores.append(loss - DEL_WT * deletedChars
                              - ONE_LETTER_WT * oneLetter)
                segSamples.append(segs)
                dels.append(deletedChars)

            segProbs, bestSegs = guessSegTargets(scores, segSamples, pSegs,
                                                 metric=METRIC)

            Xb, deleted, oneLetter = segsToX(utts, bestSegs,
                                             maxutt, maxlen, ctable)
            if reverseUtt:
                yb = Xb[:, ::-1, :]
            else:
                yb = Xb

            allBestSegs[inds] = bestSegs

            loss = model.train_on_batch(Xb, yb)
            segmenter.train_on_batch(XCb, np.expand_dims(segProbs, 2))
            epochLoss += loss[0]
            epochDel += deleted.sum()
            epochOneL += oneLetter.sum()

            # if batch % 25 == 0:
            #     print("Loss:", loss)
            #     print("Mean deletions:", np.array(dels).sum(axis=1).mean())
            #     print("Deletions in best:", deleted.sum())

            if printSome:
                toPrint = 10

                predLst = []

                for smp in range(nSamples):
                    segs = segSamples[smp]
                    Xb, deleted,oneLetter = segsToX(utts, segs,
                                                    maxutt, maxlen, ctable)
                    if reverseUtt:
                        yb = Xb[:, ::-1, :]
                    else:
                        yb = Xb

                    preds = model.predict(Xb[:toPrint], verbose=0)
                    if reverseUtt:
                        preds = preds[:, ::-1, :]
                    predLst.append(preds)

                #augment everything with the post-computed "best"
                segs = bestSegs
                Xb, deleted, oneLetter = segsToX(utts, segs,
                                                 maxutt, maxlen, ctable)
                if reverseUtt:
                    yb = Xb[:, ::-1, :]
                else:
                    yb = Xb

                preds = model.predict(Xb[:toPrint], verbose=0)
                if reverseUtt:
                    preds = preds[:, ::-1, :]
                predLst.append(preds)
                segSamples.append(bestSegs)
                scores.append(lossByUtt(model, Xb, yb, BATCH_SIZE,))

                for utt in range(toPrint):
                    for smp in [nSamples,]: #range(N_SAMPLES + 1):
                        #print(utts[utt])

                        thisSeg = segSamples[smp][utt]
                        rText = reconstruct(utts[utt], thisSeg, maxutt)

                        print(realize(rText, maxlen, maxutt))

                        for wi in range(maxutt):
                            guess = ctable.decode(
                                predLst[smp][utt, wi], calc_argmax=True)
                            print(guess, end=" ")
                        print()
                        print("Score", scores[smp][utt], "del", deleted[utt])
                print()

        print("Loss:", epochLoss)
        print("Deletions:", epochDel)
        print("One letter words:", epochOneL)
        printSegScore(text, allBestSegs)
        writeLog(iteration, epochLoss, epochDel, 
                 text, allBestSegs, logfile)

        if iteration % 10 == 0:
            writeSolutions(logdir, model, segmenter,
                           allBestSegs, text, iteration)

    print("Logs in", logdir)
