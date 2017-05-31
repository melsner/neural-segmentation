from __future__ import print_function, division
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Model, Sequential, load_model
try:
    from keras.engine.training import slice_X
except:
    from keras.engine.training import _slice_arrays as slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Input, Reshape, Merge, merge, Lambda, Dropout, Masking
from keras import backend as K
import tensorflow as tf
from tensorflow.python.platform.test import is_gpu_available
import numpy as np
import numpy.ma as ma
import cPickle as pickle
import random
import sys
import re
import copy
import time
from itertools import izip
from collections import defaultdict
from echo_words import CharacterTable, pad
from capacityStatistics import getPseudowords
from ae_io import *
from data_handling import *
from sampling import *
from scoring import *

argmax = lambda array: max(izip(array, xrange(len(array))))[1]
argmin = lambda array: min(izip(array, xrange(len(array))))[1]

def inputSliceClosure(dim):
    return lambda xx: xx[:, dim, :, :]

def outputSliceClosure(dim):
    return lambda xx: xx[:, dim, :]

def mask_output(x, input, mask_value, reverseUtt):
    if reverseUtt:
        m = K.any(K.not_equal(K.reverse(input, 1), mask_value), axis=-1, keepdims=True)
    else:
        m = K.any(K.not_equal(input, mask_value), axis=-1, keepdims=True)
    x *= K.cast(m, 'float32')
    x += 1 - K.cast(m, 'float32')
    return x


def masked_categorical_crossentropy(y_true, y_pred):
    mask_base = K.any(y_true, axis=-1, keepdims=True)
    mask = K.cast(mask_base, 'float32')
    y_pred *= mask
    y_true *= mask
    y_pred += 1-mask
    y_pred += 1-mask
    losses = K.categorical_crossentropy(y_pred, y_true)
    losses *= K.squeeze(mask, -1) # / K.mean(mask)
    return losses

def masked_mean_squared_error(y_true, y_pred):
    y_pred = y_pred * K.cast(K.any(K.reverse(y_true, 1), axis=-1, keepdims=True), 'float32')
    return K.mean(K.square(y_pred - y_true), axis=-1)

def masked_categorical_accuracy(y_true, y_pred):
    mask = K.cast(K.any(y_true, axis=-1, keepdims=True), 'float32')
    accuracy = K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())
    return accuracy * K.squeeze(mask, -1) # / K.mean(mask)

def trainAEOnly(ae, Xs, Xs_mask, segs, trainIters, batch_size, logdir, reverseUtt, acoustic):
    print('Training auto-encoder network')

    ## Preprocess input data
    Xae, deletedChars, oneLetter = XsSeg2Xae(Xs,
                                             Xs_mask,
                                             segs,
                                             maxUtt,
                                             maxLen,
                                             acoustic)

    ## Randomly permute samples
    p, p_inv = getRandomPermutation(len(Xae))
    Xae = Xae[p]

    Yae = getYae(Xae, reverseUtt, acoustic)

    ## Re-initialize network weights in case any training has already happened
    ae.load_weights(logdir + '/model.h5', by_name=True)

    ae.fit(Xae,
           Yae,
           batch_size=batch_size,
           epochs=trainIters)

    return Xae[p_inv]

def trainSegmenterOnly(segmenter, Xs, Xs_mask, Y, trainIters, batch_size, logdir):
    print('Training segmenter network')

    segsProposal = dict.fromkeys(Xs.keys())
    segScores = dict.fromkeys(Xs.keys())

    ## Re-initialize network weights in case any training has already happened
    segmenter.load_weights(logdir + '/segmenter.h5', by_name=True)

    ## Randomly permute samples
    p, p_inv = getRandomPermutation(len(Xs))
    Xs = Xs[p]
    Xs_mask = Xs_mask[p]

    segmenter.fit(Xs,
                  Y,
                  batch_size=batch_size,
                  epochs=trainIters)

    Xs = Xs[p_inv]
    Xs_mask = Xs_mask[p_inv]

    print('Getting model predictions for evaluation')
    for doc in segsProposal:
        masked_prediction = np.ma.array(segmenter.predict(Xs[doc], batch_size=batch_size) > 0.5, mask=Xs_mask[doc])
        segsProposal[doc] = masked_prediction.compressed()

    return segsProposal

if __name__ == "__main__":

    ## Process CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("dataDir")
    parser.add_argument("--acoustic", action='store_true')
    parser.add_argument("--noSegNet", action='store_true')
    parser.add_argument("--supervisedSegmenter", action='store_true')
    parser.add_argument("--supervisedAE", action='store_true')
    parser.add_argument("--reverseUtt", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--algorithm", type=str, choices=['1best', 'importance', 'viterbi'], default='importance')
    parser.add_argument("--wordHidden")
    parser.add_argument("--uttHidden")
    parser.add_argument("--segHidden")
    parser.add_argument("--wordDropout")
    parser.add_argument("--charDropout")
    parser.add_argument("--metric")
    parser.add_argument("--pretrainIters")
    parser.add_argument("--trainNoSegIters")
    parser.add_argument("--trainIters")
    parser.add_argument("--maxChar")
    parser.add_argument("--maxUtt")
    parser.add_argument("--maxLen")
    parser.add_argument("--delWt")
    parser.add_argument("--oneLetterWt")
    parser.add_argument("--segWt")
    parser.add_argument("--nSamples")
    parser.add_argument("--batchSize")
    parser.add_argument("--samplingBatchSize")
    parser.add_argument("--logfile")
    parser.add_argument("--gpufrac", default=0.15)
    args = parser.parse_args()
    try:
        args.gpufrac = float(args.gpufrac)
    except:
        args.gpufrac = None

    ## GPU management stuff
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpufrac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    usingGPU = is_gpu_available()
    print('Using GPU: %s' %usingGPU)
    K.set_session(sess)
    K.set_learning_phase(1)

    ## Load any saved data/parameters
    t0 = time.time()
    print()
    print('Loading data...')
    print()

    load_models = False
    if args.logfile == None:
        logdir = "logs/" + str(os.getpid()) + '/'
    else:
        logdir = "logs/" + args.logfile

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        if not (args.supervisedAE or args.supervisedSegmenter):
            load_models = True

    if load_models and os.path.exists(logdir + '/checkpoint.obj'):
        print('Training checkpoint data found. Loading...')
        with open(logdir + '/checkpoint.obj', 'rb') as f:
            checkpoint = pickle.load(f)
    else:
        print('No training checkpoint found. Starting training from beginning.')
        checkpoint = {}

    ## Initialize system parameters
    dataDir = args.dataDir
    ACOUSTIC = checkpoint.get('acoustic', args.acoustic)
    SEG_NET = checkpoint.get('segNet', not args.noSegNet)
    ALGORITHM = checkpoint.get('algorithm', args.algorithm)
    REVERSE_UTT = checkpoint.get('reverseUtt', args.reverseUtt)
    MASK_VALUE = 0 if ACOUSTIC else 1
    wordHidden = checkpoint.get('wordHidden', int(args.wordHidden) if args.wordHidden else 100 if ACOUSTIC else 40)
    uttHidden = checkpoint.get('uttHidden', int(args.uttHidden) if args.uttHidden else 500 if ACOUSTIC else 400)
    segHidden = checkpoint.get('segHidden', int(args.segHidden) if args.segHidden else 500 if ACOUSTIC else 100)
    wordDropout = checkpoint.get('wordDropout', float(args.wordDropout) if args.wordDropout else 0.25 if ACOUSTIC else 0.25)
    charDropout = checkpoint.get('charDropout', float(args.charDropout) if args.charDropout else 0.25 if ACOUSTIC else 0.25)
    maxChar = checkpoint.get('maxChar', int(args.maxChar) if args.maxChar else 500 if ACOUSTIC else 30)
    maxUtt = checkpoint.get('maxUtt', int(args.maxUtt) if args.maxUtt else 50 if ACOUSTIC else 10)
    maxLen = checkpoint.get('maxLen', int(args.maxLen) if args.maxLen else 100 if ACOUSTIC else 7)
    pretrainIters = int(args.pretrainIters) if args.pretrainIters else 10 if ACOUSTIC else 10
    trainNoSegIters = int(args.trainNoSegIters) if args.trainNoSegIters else 10 if ACOUSTIC else 10
    trainIters = int(args.trainIters) if args.trainIters else 100 if ACOUSTIC else 80
    METRIC = checkpoint.get('metric', args.metric if args.metric else 'mse' if ACOUSTIC else 'logprob')
    DEL_WT = checkpoint.get('delWt', float(args.delWt) if args.delWt else 1 if ACOUSTIC else 50)
    ONE_LETTER_WT = checkpoint.get('oneLetterWt', float(args.oneLetterWt) if args.oneLetterWt else 50 if ACOUSTIC else 10)
    SEG_WT = checkpoint.get('segWt', float(args.segWt) if args.segWt else 1 if ACOUSTIC else 1)
    N_SAMPLES = checkpoint.get('nSamples', int(args.nSamples) if args.nSamples else 50 if ACOUSTIC else 50)
    BATCH_SIZE = checkpoint.get('batchSize', int(args.batchSize) if args.batchSize else 128 if ACOUSTIC else 128)
    SAMPLING_BATCH_SIZE = checkpoint.get('samplingBatchSize', int(args.samplingBatchSize) if args.samplingBatchSize else 128 if ACOUSTIC else 128)
    iteration = checkpoint.get('iteration', 0)
    pretrain = checkpoint.get('pretrain', True)
    DEBUG = args.debug
    wordDecLayers = 1
    RNN = recurrent.LSTM

    checkpoint['acoustic'] = ACOUSTIC
    checkpoint['segNet'] = SEG_NET
    checkpoint['algorithm'] = ALGORITHM
    checkpoint['reverseUtt'] = REVERSE_UTT
    checkpoint['wordHidden'] = wordHidden
    checkpoint['uttHidden'] = uttHidden
    checkpoint['segHidden'] = segHidden
    checkpoint['wordDropout'] = wordDropout
    checkpoint['charDropout'] = charDropout
    checkpoint['maxChar'] = maxChar
    checkpoint['maxUtt'] = maxUtt
    checkpoint['maxLen'] = maxLen
    checkpoint['pretrainIters'] = pretrainIters
    checkpoint['trainNoSegIters'] = trainNoSegIters
    checkpoint['trainIters'] = trainIters
    checkpoint['metric'] = METRIC
    checkpoint['delWt'] = DEL_WT
    checkpoint['oneLetterWt'] = ONE_LETTER_WT
    checkpoint['segWt'] = SEG_WT
    checkpoint['nSamples'] = N_SAMPLES
    checkpoint['batchSize'] = BATCH_SIZE
    checkpoint['samplingBatchSize'] = SAMPLING_BATCH_SIZE
    checkpoint['iteration'] = iteration
    checkpoint['pretrain'] = pretrain

    ## Load training data
    if ACOUSTIC:
        segfile_paths = [dataDir + x for x in os.listdir(dataDir) if x.endswith('_vad.txt')]
        goldwrd_paths = [dataDir + x for x in os.listdir(dataDir) if x.endswith('.wrd')]
        goldphn_paths = [dataDir + x for x in os.listdir(dataDir) if x.endswith('.phn')]
        SEGFILE = segfile_paths[0] if len(segfile_paths) > 0 else None
        GOLDWRD = goldwrd_paths[0] if len(goldwrd_paths) > 0 else None
        GOLDPHN = goldphn_paths[0] if len(goldphn_paths) > 0 else None
        assert SEGFILE and (GOLDWRD or GOLDPHN), \
            'Files containing initial and gold segmentations are required in acoustic mode.'
        if 'intervals' in checkpoint and 'segs_init' in checkpoint:
            intervals, segs_init = checkpoint['intervals'], checkpoint['segs_init']
        else:
            intervals, segs_init = timeSegs2frameSegs(SEGFILE)
            checkpoint['intervals'] = intervals
            checkpoint['segs_init'] = segs_init
        vadBreaks = checkpoint.get('vadBreaks', intervals2ForcedSeg(intervals))
        checkpoint['vadBreaks'] = vadBreaks
        if 'gold' in checkpoint:
            gold = checkpoint['gold']
        else:
            gold = {'wrd': None, 'phn': None}
            if GOLDWRD:
                gold['wrd'] = readGoldFrameSeg(GOLDWRD)
            if GOLDPHN:
                gold['phn'] = readGoldFrameSeg(GOLDPHN)
            checkpoint['gold'] = gold
        if GOLDWRD:
            goldWrdSegCts = 0
            for doc in gold['wrd']:
                goldWrdSegCts += len(gold['wrd'][doc])
            print('Gold word segmentation count: %d' %goldWrdSegCts)
        if GOLDPHN:
            goldPhnSegCts = 0
            for doc in gold['phn']:
                goldPhnSegCts += len(gold['phn'][doc])
            print('Gold phone segmentation count: %d' % goldPhnSegCts)
        if 'raw' in checkpoint:
            raw = checkpoint['raw']
            FRAME_SIZE = raw[raw.keys()[0]].shape[-1]
        else:
            raw, FRAME_SIZE = readMFCCs(dataDir)
            raw = filterMFCCs(raw, intervals, segs_init, FRAME_SIZE)
            checkpoint['raw'] = raw

        if False:  # text['wrd'] and not args.supervisedSegmenter:
            print('Initial word segmentation scores:')
            printSegScores(getSegScores(gold['wrd'], frameSegs2timeSegs(intervals, segs_init), ACOUSTIC), True)
            print()
        if False:  # text['phn'] and not args.supervisedSegmenter:
            print('Initial phone segmentation scores:')
            printSegScores(getSegScores(gold['phn'], frameSegs2timeSegs(intervals, segs_init), ACOUSTIC), True)
            print()

    else:
        if 'gold' in checkpoint and 'raw' in checkpoint and 'charset' in checkpoint:
            gold, raw, charset = checkpoint['gold'], checkpoint['raw'], checkpoint['charset']
        else:
            gold, raw, charset = readTexts(dataDir)
        print('Corpus length (words):', sum([len(w) for d in gold for w in gold[d]]))
        print('Corpus length (characters):', sum([len(u) for d in raw for u in raw[d]]))
        ctable = CharacterTable(charset)

    doc_list = sorted(list(raw.keys()))
    segsProposal = checkpoint.get('segsProposal', dict.fromkeys(doc_list))
    deletedChars = checkpoint.get('deletedChars', dict.fromkeys(doc_list))
    oneLetter = checkpoint.get('oneLetter', dict.fromkeys(doc_list))

    checkpoint['segsProposal'] = segsProposal
    checkpoint['deletedChars'] = deletedChars
    checkpoint['oneLetter'] = oneLetter

    charDim = FRAME_SIZE if ACOUSTIC else len(charset)
    doc_list = sorted(list(raw.keys()))
    raw_cts = {}
    for doc in raw:
        if ACOUSTIC:
            raw_cts[doc] = raw[doc].shape[0]
        else:
            raw_cts[doc] = sum([len(utt) for utt in raw[doc]])
    raw_total = sum([raw_cts[doc] for doc in raw_cts])
    ## Xs: segmenter input (unsegmented input sequences by utterance)
    Xs, doc_indices = frameInputs2Utts(raw, vadBreaks, maxChar) if ACOUSTIC else texts2Xs(raw, maxChar, ctable)
    if DEBUG and not ACOUSTIC:
        n = 20
        print('Character reconstruction check:')
        for doc in doc_list:
            s = doc_indices[doc][0]
            print('Document: %s' %doc)
            reconstruction = reconstructXs(Xs[s:n], ctable)
            for i in range(n):
                print('Input string:   %s' %raw[doc][i])
                print('Reconstruction: %s' %reconstruction[i])
    ## Xs_mask: mask of padding timesteps by utterance
    if ACOUSTIC:
        Xs_mask = getMask(Xs)
    else:
        Xs_mask = np.zeros((len(Xs), maxChar))
        for doc in doc_list:
            s,e = doc_indices[doc]
            Xs_mask_doc = np.zeros((e-s, maxChar))
            for i in range(len(raw[doc])):
                utt_len = len(raw[doc][i])
                Xs_mask_doc[i][utt_len:] = 1
            Xs_mask[s:e] = Xs_mask_doc

    ## pSegs: segmentation proposal distribution
    if ACOUSTIC:
        pSegs = segs2pSegsWithForced(segs_init, vadBreaks, alpha=0.05)
        pSegs = frameSegs2FrameSegsXUtt(pSegs, vadBreaks, maxChar, doc_indices)
        vad = frameSegs2FrameSegsXUtt(vadBreaks, vadBreaks, maxChar, doc_indices)
    else:
        pSegs = 0.2 * np.ones((len(Xs), maxChar, 1))
        pSegs[:, 0] = 1.
    ## Zero-out segmentation probability in padding regions
    pSegs[np.where(Xs_mask)] = 0.
    ## Data loading finished, save checkpoint
    with open(logdir + '/checkpoint.obj', 'ab') as f:
        checkpoint['iteration'] = iteration
        pickle.dump(checkpoint, f)

    t1 = time.time()
    print('Data loaded in %ds.' % (t1 - t0))
    print()

    ## Log system parameters
    with open(logdir + '/params.txt', 'wb') as f:
        print('Model parameters:', file=f)
        if ACOUSTIC:
            print('  Input type: Acoustic', file=f)
            if SEGFILE:
                print('  Initial segmentation file: %s' % SEGFILE, file=f)
            if GOLDWRD:
                print('  Gold word segmentation file: %s' % GOLDWRD, file=f)
            if GOLDPHN:
                print('  Gold phoneme segmentation file: %s' % GOLDPHN, file=f)
        else:
            print('  Input type: Text', file=f)
        print('  Using segmenter network: %s' % SEG_NET, file=f)
        print('  Input data location: %s' % dataDir, file=f)
        print('  Autoencoder loss function: %s' % METRIC, file=f)
        print('  Word layer hidden units: %s' % wordHidden, file=f)
        print('  Utterance layer hidden units: %s' % uttHidden, file=f)
        print('  Segmenter network hidden units: %s' % segHidden, file=f)
        print('  Word dropout rate: %s' % wordDropout, file=f)
        print('  Character dropout rate: %s' % charDropout, file=f)
        print('  Maximum utterance length (characters): %s' % maxLen, file=f)
        print('  Maximum utterance length (words): %s' % maxUtt, file=f)
        print('  Maximum word length (characters): %s' % maxChar, file=f)
        print('  Deletion penalty: %s' % DEL_WT, file=f)
        print('  One letter segment penalty: %s' % ONE_LETTER_WT, file=f)
        print('  Segmentation penalty: %s' % SEG_WT, file=f)
        print('  Number of samples per batch: %s' % N_SAMPLES, file=f)
        print('  Batch size: %s' % BATCH_SIZE, file=f)
        print('  Pretraining iterations: %s' % pretrainIters, file=f)
        print('  Training iterations without segmenter network: %s' % trainNoSegIters, file=f)
        print('  Training iterations (total): %s' % trainIters, file=f)
        print('', file=f)
        print('Command line call to repro/resume:', file=f)
        print('', file=f)
        print('python scripts/main.py',
              '%s' % dataDir,
              '--acoustic' if ACOUSTIC else '',
              '--noSegNet' if not SEG_NET else '',
              '--reverseUtt' if REVERSE_UTT else '',
              '--algorithm %s' % ALGORITHM,
              '--wordHidden %s' % wordHidden,
              '--uttHidden %s' % uttHidden,
              '--segHidden %s' % segHidden,
              '--wordDropout %s' % wordDropout,
              '--charDropout %s' % charDropout,
              '--metric %s' % METRIC,
              '--pretrainIters %s' % pretrainIters,
              '--trainNoSegIters %s' % trainNoSegIters,
              '--trainIters %s' % trainIters,
              '--maxChar %s' % maxChar,
              '--maxUtt %s' % maxUtt,
              '--maxLen %s' % maxLen,
              '--delWt %s' % DEL_WT,
              '--oneLetterWt %s' % ONE_LETTER_WT,
              '--segWt %s' % SEG_WT,
              '--nSamples %s' % N_SAMPLES,
              '--batchSize %s' % BATCH_SIZE,
              '--samplingBatchSize %s' % BATCH_SIZE,
              '--logfile %s' % logdir[5:],
              '--gpufrac %s' % args.gpufrac, file=f)

    print("Logging at", logdir)

    print("Constructing networks.")

    ## CONSTRUCT NETWORK GRAPHS
    ## 1. Auto-encoder
    ## a. Word-level AE
    if True:
        wordEncoder = Sequential()
        wordEncoder.add(Dropout(charDropout, input_shape=(maxLen, charDim),
                                noise_shape=(1, maxLen, 1)))
        wordEncoder.add(RNN(wordHidden, name="encoder-rnn"))

        wordDecoder = Sequential()
        wordDecoder.add(RepeatVector(input_shape=(wordHidden,),
                                     n=maxLen, name="encoding"))
        for ii in range(wordDecLayers):
            wordDecoder.add(RNN(wordHidden, return_sequences=True,
                                name="decoder%i" % ii))

        wordDecoder.add(TimeDistributed(Dense(charDim, name="dense"), name="td"))
        if ACOUSTIC:
            wordDecoder.add(Activation('linear', name='linear'))
        else:
            wordDecoder.add(Activation('softmax', name="softmax"))

        inp = Input(shape=(maxUtt, maxLen, charDim))
        resh = TimeDistributed(wordEncoder)(inp)

        resh = TimeDistributed(Dropout(wordDropout, noise_shape=(1,)))(resh)

        ## b. Utterance-level AE
        encoderLSTM = RNN(uttHidden, name="utt-encoder")(resh)
        repeater = RepeatVector(maxUtt, input_shape=(wordHidden,))(encoderLSTM)
        decoder = RNN(uttHidden, return_sequences=True,
                      name="utt-decoder")(repeater)
        decOut = TimeDistributed(Dense(wordHidden, activation="linear"))(decoder)

        resh2 = TimeDistributed(wordDecoder)(decOut)

        ## c. Output masking (words)
        if REVERSE_UTT:
            mask = Lambda(lambda x: x * K.cast(K.any(K.reverse(inp, 1), -1, keepdims=True), 'float32'))(resh2)
        else:
            mask = Lambda(lambda x: x * K.cast(K.any(inp, -1, keepdims=True), 'float32'))(resh2)
        mask = Masking(mask_value=0)(mask)

        ## d. Model inputs/targets
        model = Model(input=inp, output=mask)

    if ACOUSTIC:
        model.compile(loss="mean_squared_error",
                      optimizer='adam')
    else:
        model.compile(loss="categorical_crossentropy",
                      optimizer='adam',
                      metrics=["accuracy"])
    model.summary()

    ## 2. Segmenter
    if SEG_NET:
        segmenter = Sequential()
        segmenter.add(Masking(mask_value=0.0, input_shape=(maxChar, charDim)))
        segmenter.add(RNN(segHidden, return_sequences=True, name="segmenter"))
        segmenter.add(TimeDistributed(Dense(1)))
        segmenter.add(Activation("sigmoid"))
        segmenter.compile(loss="binary_crossentropy",
                          optimizer="adam")
        segmenter.summary()

    if load_models and os.path.exists(logdir + '/model.h5'):
        print('Autoencoder checkpoint found. Loading weights...')
        model.load_weights(logdir + '/model.h5', by_name=True)
    else:
        print('No autoencoder checkpoint found. Keeping default initialization.')
        model.save(logdir + '/model.h5')
    if SEG_NET:
        if load_models and os.path.exists(logdir + '/segmenter.h5'):
            print('Segmenter checkpoint found. Loading weights...')
            segmenter.load_weights(logdir + '/segmenter.h5', by_name=True)
        else:
            print('No segmenter checkpoint found. Keeping default initialization.')
            segmenter.save(logdir + '/segmenter.h5')

    ## Unit testing of segmenter network by training on gold segmentations
    if args.supervisedAE or args.supervisedSegmenter:
        print('')
        print('Supervised network test (word-level segmentations)')

        ## Gold segmentations
        print('Using gold segmentations')
        if ACOUSTIC:
            _, goldseg = timeSegs2frameSegs(GOLDWRD)
            Y = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
        else:
            goldseg = gold
            Y = texts2Segs(gold, maxChar)
            #scores = getSegScores(gold, Y, acoustic=ACOUSTIC)
            #printSegScores(scores, acoustic=ACOUSTIC)

        if args.supervisedAE and False:
            Xae = trainAEOnly(model,
                              Xs,
                              Xs_mask,
                              Y,
                              trainIters,
                              BATCH_SIZE,
                              logdir,
                              REVERSE_UTT,
                              ACOUSTIC)

            if not ACOUSTIC:
                printReconstruction(10, model, Xae, ctable, BATCH_SIZE, REVERSE_UTT)

        if args.supervisedSegmenter:
            segsProposal = trainSegmenterOnly(segmenter,
                                              Xs,
                                              Xs_mask,
                                              Y,
                                              trainIters,
                                              BATCH_SIZE,
                                              logdir)
            print('Scoring segmentations')
            if ACOUSTIC:
                scores = getSegScores(gold['wrd'], frameSegs2timeSegs(intervals, segsProposal), acoustic=ACOUSTIC)
            else:
                scores = getSegScores(gold, segsProposal, acoustic=ACOUSTIC)
            printSegScores(scores, acoustic=ACOUSTIC)

        ## Random segmentations
        print('Using random segmentations')
        Y = sampleSeg(pSegs)
        if args.supervisedAE:
            Xae = trainAEOnly(model,
                              Xs,
                              Xs_mask,
                              Y,
                              trainIters,
                              BATCH_SIZE,
                              logdir,
                              REVERSE_UTT,
                              ACOUSTIC)

            if not ACOUSTIC:
                printReconstruction(10, model, Xae, ctable, BATCH_SIZE, REVERSE_UTT)

        if args.supervisedSegmenter:
            segsProposal = trainSegmenterOnly(segmenter,
                                              Xs,
                                              Xs_mask,
                                              Y,
                                              trainIters,
                                              BATCH_SIZE,
                                              logdir)
            print('Scoring segmentations')
            if ACOUSTIC:
                scores = getSegScores(gold['wrd'], frameSegs2timeSegs(intervals, segsProposal), acoustic=ACOUSTIC)
            else:
                scores = getSegScores(gold, segsProposal, acoustic=ACOUSTIC)
            printSegScores(scores, acoustic=ACOUSTIC)

        if ACOUSTIC:
            print('Supervised network test (phone-level segmentations')

            ## Gold segmentations
            print('Using gold segmentations')
            _, goldseg = timeSegs2frameSegs(GOLDPHN)
            Y = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)

            if args.supervisedAE:
                Xae = trainAEOnly(model,
                                  Xs,
                                  Xs_mask,
                                  Y,
                                  trainIters,
                                  BATCH_SIZE,
                                  logdir,
                                  REVERSE_UTT,
                                  ACOUSTIC)
            if not ACOUSTIC:
                printReconstruction(10, model, Xae, ctable, BATCH_SIZE, REVERSE_UTT)

            if args.supervisedSegmenter:
                segsProposal = trainSegmenterOnly(segmenter,
                                                  Xs,
                                                  Xs_mask,
                                                  Y,
                                                  trainIters,
                                                  BATCH_SIZE,
                                                  logdir)
                print('Scoring segmentations')
                scores = getSegScores(gold['phn'], frameSegs2timeSegs(intervals, segsProposal), acoustic=ACOUSTIC)
                printSegScores(scores, acoustic=ACOUSTIC)

            ## Random segmentations
            print('Using random segmentations')
            Y = sampleSeg(pSegs)
            if args.supervisedAE:
                Xae = trainAEOnly(model,
                                  Xs,
                                  Xs_mask,
                                  Y,
                                  trainIters,
                                  BATCH_SIZE,
                                  logdir,
                                  REVERSE_UTT,
                                  ACOUSTIC)
            if not ACOUSTIC:
                printReconstruction(10, model, Xae, ctable, BATCH_SIZE, REVERSE_UTT)

            if args.supervisedSegmenter:
                segsProposal = trainSegmenterOnly(segmenter,
                                                  Xs,
                                                  Xs_mask,
                                                  Y,
                                                  trainIters,
                                                  BATCH_SIZE,
                                                  logdir)
                print('Scoring segmentations')
                scores = getSegScores(gold['phn'], frameSegs2timeSegs(intervals, segsProposal), acoustic=ACOUSTIC)
                printSegScores(scores, acoustic=ACOUSTIC)

        exit()





    ## Auto-encoder pretraining on initial segmentation proposal
    print()
    print("Pre-training network")
    while iteration < pretrainIters and pretrain:
        print('-' * 50)
        print('Iteration', iteration + 1)

        t0 = time.time()
        segs = sampleSeg(pSegs)
        ts0 = time.time()
        Xae, deletedChars, oneLetter = XsSeg2Xae(Xs,
                                                 Xs_mask,
                                                 segs,
                                                 maxUtt,
                                                 maxLen,
                                                 ACOUSTIC)
        ts1 = time.time()
        print('Split time: %.2fs' %(ts1-ts0))

        ## Randomly permute samples
        p, p_inv = getRandomPermutation(len(Xae))
        Xae = Xae[p]
        Xs = Xs[p]

        if DEBUG and not ACOUSTIC:
            n = 5
            print('Character reconstruction check:')
            for doc in doc_list:
                print('Document: %s' % doc)
                s = doc_indices[doc][0]
                reconstructionXs = reconstructXs(Xs[s:s+n], ctable)
                reconstructionXae = reconstructXae(Xae[s:s+n], ctable)
                for i in range(n):
                    print('Segmentation points: %s' % ' '.join([str(ix) for ix in range(len(segs[s+i])) \
                                                                if int(segs[s+i,ix,0]) == 1]))
                    print('Input tensor:        %s' % reconstructionXs[i])
                    print('Reconstruction:      %s' % reconstructionXae[i])
                    print()
        #print(Xae[Xae.keys()[0]].shape)
        #print(Xae[Xae.keys()[0]][:5].argmax(-1))
        #print(reconstructXae(Xae[Xae.keys()[0]][:5], ctable))
        #exit()

        t1 = time.time()
        print('Auto-encoder input preprocessing completed in %.2fs.' %(t1-t0))

        Yae = getYae(Xae, REVERSE_UTT, ACOUSTIC)

        if DEBUG and False:
            preds = model.predict(Xae, batch_size=BATCH_SIZE)
            print('Finite prediction cells: %s' %np.isfinite(preds).sum())

        print("Total deleted chars:    %d" %(deletedChars.sum()))
        print("Total one-letter words: %d" %(oneLetter.sum()))

        if False:
            print('Training segmenter network on random segmentation.')
            segmenter.fit(np.concatenate([Xs[d] for d in Xs]),
                          np.concatenate([segs[d] for d in segs]),
                          batch_size=BATCH_SIZE,
                          epochs=1)

        Xae = Xae[p_inv]
        Xs = Xs[p_inv]

        print('Training auto-encoder network on random segmentation.')
        model.fit(Xae,
                  Yae,
                  batch_size=BATCH_SIZE,
                  epochs=1)

        # Correctness checks for NN masking
        if DEBUG:
            out = model.predict(Xae, batch_size=BATCH_SIZE)
            print('Timesteps in input: %s' %Xae.any(-1).sum())
            print('Timesteps in output: %s (should equal timesteps in input)' %out.any(-1).sum())
            print('Deleted timesteps: %s' %int(deletedChars.sum()))
            print('Timesteps + deleted: %s (should be %s)' % (out.any(-1).sum() + int(deletedChars.sum()), sum([raw_cts[doc] for doc in raw_cts])))
            print('')

        iteration += 1
        if iteration == pretrainIters:
            pretrain = False
            iteration = 0

        model.save(logdir + '/model.h5')
        with open(logdir + '/checkpoint.obj', 'wb') as f:
            checkpoint['iteration'] = iteration
            checkpoint['pretrain'] = pretrain
            pickle.dump(checkpoint, f)

        if not ACOUSTIC:
            preds = model.predict(Xae[:10])
            printReconstruction(10, model, Xae, ctable, BATCH_SIZE, REVERSE_UTT)

    ## Pretraining finished, update checkpoint
    with open(logdir + '/checkpoint.obj', 'wb') as f:
        checkpoint['pretrain'] = pretrain
        checkpoint['iteration'] = iteration
        pickle.dump(checkpoint, f)

    segScores = {'wrd': dict.fromkeys(doc_list), 'phn': dict.fromkeys(doc_list)}
    segScores['wrd']['##overall##'] = [(0,0,0), None, (0,0,0), None]
    segScores['phn']['##overall##'] = [(0,0,0), None, (0,0,0), None]
    print()

    N_BATCHES = math.ceil(len(Xs)/SAMPLING_BATCH_SIZE)

    print('Training networks jointly')
    while iteration < trainIters:

        it0 = time.time()

        print()
        print('-' * 50)
        print('Iteration', iteration + 1)

        if SEG_NET:
            if iteration < trainNoSegIters:
                print('Using initialization as proposal distribution.')
            else:
                print('Using segmenter network as proposal distribution.')
        else:
            if iteration == 0:
                print('Using initialization as proposal distribution.')
            else:
                print('Using sampled probabilities as proposal distribution.')

        segsProposal = []
        deletedChars = []
        oneLetter = []

        epochLoss = 0
        if not ACOUSTIC:
            epochAcc = 0
        epochDel = 0
        epochOneL = 0
        epochSeg = 0

        ## Randomly permute samples
        p, p_inv = getRandomPermutation(len(Xs))
        Xs = Xs[p]
        Xs_mask = Xs_mask[p]
        pSegs = pSegs[p]
        if ACOUSTIC:
            vad = vad[p]

        for b in range(0, len(Xs), SAMPLING_BATCH_SIZE):

            bt0 = time.time()

            Xs_batch = Xs[b:b+SAMPLING_BATCH_SIZE]
            Xs_mask_batch = Xs_mask[b:b+SAMPLING_BATCH_SIZE]
            if ACOUSTIC:
                vad_batch = vad[b:b+SAMPLING_BATCH_SIZE]

            if iteration < trainNoSegIters:
                pSegs_batch = pSegs[b:b+SAMPLING_BATCH_SIZE]
            else:
                if SEG_NET:
                    preds = segmenter.predict(Xs_batch, batch_size = BATCH_SIZE)
                    ## Interpolate with sampled distribution at 5:5
                    #pSegs_batch = .5 * preds + .5 * pSegs[b:b + SAMPLING_BATCH_SIZE]
                    ## Interpolate with uniform distribution at 9:1
                    pSegs_batch = .9 * preds + .1 * .5 * np.ones(preds.shape)
                else:
                    pSegs_batch = pSegs[b:b + SAMPLING_BATCH_SIZE]
                    ## Interpolate with uniform distribution at 9:1
                    pSegs_batch = .9 * pSegs_batch + .1 * .5 * np.ones(pSegs_batch.shape)

                ## Force segmentations where needed
                if ACOUSTIC:
                    pSegs_batch[np.where(vad_batch)] = 1.
                else:
                    pSegs_batch[:,0] = 1.

                ## Zero-out segmentation probability in padding regions
                pSegs_batch[np.where(Xs_mask_batch)] = 0.

            st0 = time.time()
            scores_batch = np.zeros((len(Xs_batch), N_SAMPLES, maxUtt))
            segSamples_batch = np.zeros((len(Xs_batch), N_SAMPLES, maxChar))
            print()
            print('Batch %d/%d' %((b+1)/SAMPLING_BATCH_SIZE+1, N_BATCHES))
            for s in range(N_SAMPLES):
                sys.stdout.write('\rSample %d/%d' %(s+1, N_SAMPLES))
                sys.stdout.flush()
                segs_batch = sampleSeg(pSegs_batch)
                segSamples_batch[:,s,:] = np.squeeze(segs_batch, -1)

                Xae_batch, deletedChars_batch, oneLetter_batch = XsSeg2Xae(Xs_batch,
                                                                           Xs_mask_batch,
                                                                           segs_batch,
                                                                           maxUtt,
                                                                           maxLen,
                                                                           ACOUSTIC)

                Yae_batch = getYae(Xae_batch, REVERSE_UTT, ACOUSTIC)

                scores_batch[:,s,:] = scoreXUtt(model, Xae_batch, Yae_batch, BATCH_SIZE, metric = METRIC)
                if ALGORITHM != 'viterbi':
                    scores_batch[:,s,:] -= deletedChars_batch * DEL_WT
                    scores_batch[:,s,:] -= (oneLetter_batch / maxUtt)[:, None] * ONE_LETTER_WT
                    scores_batch[:,s,:] -= (np.squeeze(segs_batch, -1).sum(-1, keepdims=True) / maxUtt) * SEG_WT

            print('')
            print('Computing segmentation targets from samples')
            segProbs_batch, segsProposal_batch = guessSegTargets(scores_batch,
                                                                 segSamples_batch,
                                                                 pSegs_batch,
                                                                 Xs_mask_batch,
                                                                 ALGORITHM,
                                                                 maxLen,
                                                                 DEL_WT,
                                                                 ONE_LETTER_WT,
                                                                 SEG_WT)

            st1 = time.time()
            print('Sampling time: %.2fs.' %(st1-st0))

            Xae_batch, deletedChars_batch, oneLetter_batch = XsSeg2Xae(Xs_batch,
                                                                       Xs_mask_batch,
                                                                       segsProposal_batch,
                                                                       maxUtt,
                                                                       maxLen,
                                                                       ACOUSTIC)

            Yae_batch = getYae(Xae_batch, REVERSE_UTT, ACOUSTIC)

            print('Updating auto-encoder network')
            h = model.fit(Xae_batch,
                          Yae_batch,
                          batch_size=BATCH_SIZE,
                          epochs=1)

            if SEG_NET:
                print('Updating segmenter network')
                segmenter.fit(Xs_batch,
                              segProbs_batch,
                              batch_size=BATCH_SIZE,
                              epochs=1)
            else:
                pSegs[b:b + SAMPLING_BATCH_SIZE] = segProbs_batch

            epochLoss += h.history['loss'][0]
            if not ACOUSTIC:
                epochAcc += h.history['acc'][0]
            epochDel += int(deletedChars_batch.sum())
            epochOneL += int(oneLetter_batch.sum())
            epochSeg += int(segsProposal_batch.sum())

            segsProposal.append(segsProposal_batch)
            deletedChars.append(deletedChars_batch)
            oneLetter.append(oneLetter_batch)

            bt1 = time.time()
            print('Batch time: %.2fs' %(bt1-bt0))

        epochLoss /= (N_BATCHES)
        if not ACOUSTIC:
            epochAcc /= (N_BATCHES)
        segsProposal = np.concatenate(segsProposal)
        deletedChars = np.concatenate(deletedChars)
        oneLetter = np.concatenate(oneLetter)

        ## Invert random permutation so evaluation aligns correctly
        Xs = Xs[p_inv]
        Xs_mask = Xs_mask[p_inv]
        if ACOUSTIC:
            vad = vad[p_inv]
        pSegs = pSegs[p_inv]
        segsProposal = segsProposal[p_inv]
        deletedChars = deletedChars[p_inv]
        oneLetter = oneLetter[p_inv]

        iteration += 1

        model.save(logdir + '/model.h5')
        if SEG_NET:
            segmenter.save(logdir + '/segmenter.h5')
        with open(logdir + '/checkpoint.obj', 'wb') as f:
            checkpoint['iteration'] = iteration
            checkpoint['segsProposal'] = segsProposal
            checkpoint['deletedChars'] = deletedChars
            checkpoint['oneLetter'] = oneLetter
            pickle.dump(checkpoint, f)

        print('Total frames:', raw_total)
        print('Loss:', epochLoss)
        if not ACOUSTIC:
            print('Accuracy:', epochAcc)
        print('Deletions:', epochDel)
        print('One letter words:', epochOneL)
        print('Total segmentation points:', epochSeg)

        segsProposalXDoc = dict.fromkeys(doc_list)
        for doc in segsProposalXDoc:
            s,e = doc_indices[doc]
            segsProposalXDoc[doc] = segsProposal[s:e]
            if ACOUSTIC:
                masked_proposal = np.ma.array(segsProposalXDoc[doc], mask=Xs_mask[s:e])
                segsProposalXDoc[doc] = masked_proposal.compressed()

        segScore = writeLog(iteration, epochLoss, epochAcc if not ACOUSTIC else None, epochDel, epochOneL, epochSeg,
                            gold, segsProposalXDoc, logdir, intervals if ACOUSTIC else None, ACOUSTIC, print_headers=iteration == 1)


        if ACOUSTIC:
            if GOLDWRD:
                segScores['wrd']['##overall##'][1] = precision_recall_f(*segScores['wrd']['##overall##'][0])
                segScores['wrd']['##overall##'][3] = precision_recall_f(*segScores['wrd']['##overall##'][2])
                print('Word segmentation scores:')
                printSegScores(segScore['wrd'], ACOUSTIC)
            if GOLDPHN:
                segScores['phn']['##overall##'][1] = precision_recall_f(*segScores['phn']['##overall##'][0])
                segScores['phn']['##overall##'][3] = precision_recall_f(*segScores['phn']['##overall##'][2])
                print('Phone segmentation scores:')
                printSegScores(segScore['phn'], ACOUSTIC)
            writeTimeSegs(frameSegs2timeSegs(intervals, segsProposalXDoc), out_dir=logdir, TextGrid=False)
            writeTimeSegs(frameSegs2timeSegs(intervals, segsProposalXDoc), out_dir=logdir, TextGrid=True)
        else:
            printSegScores(getSegScores(gold, segsProposalXDoc, ACOUSTIC), ACOUSTIC)
            writeSolutions(logdir, segsProposalXDoc[doc_list[0]], gold[doc_list[0]], iteration)


        it1 = time.time()
        print('Iteration time: %.2fs' %(it1-it0))

    print("Logs in", logdir)

