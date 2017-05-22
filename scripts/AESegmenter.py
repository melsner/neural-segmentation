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

def trainAEOnly(ae, Xs, Xs_mask, segs, trainIters, batch_size, reverseUtt, acoustic=False):
    print('Training auto-encoder network')

    ## Preprocess input data
    Xae, deletedChars, oneLetter = XsSegs2Xae(Xs,
                                              Xs_mask,
                                              segs,
                                              maxUtt,
                                              maxLen)

    Yae = dict.fromkeys(Xae.keys())
    if reverseUtt:
        for doc in Xae:
            Yae[doc] = np.flip(Xae[doc], 1)
    else:
        Yae = Xae

    ## Reinitialize the network in case any training has already occurred
    if acoustic:
        ae.compile(loss='mean_squared_error',
                      optimizer='adam')
    else:
        ae.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    ae.fit(np.concatenate([Xae[d] for d in Xae]),
           np.concatenate([Yae[d] for d in Yae]),
           batch_size=batch_size,
           epochs=trainIters)

def trainSegmenterOnly(segmenter, Xs, Xs_mask, Y, trainIters, batch_size):
    print('Training segmenter network')

    segsProposal = dict.fromkeys(Xs.keys())
    segScores = dict.fromkeys(Xs.keys())

    ## Reinitialize the network in case any training has already occurred
    segmenter.compile(loss="binary_crossentropy",
                      optimizer="adam")

    segmenter.fit(np.concatenate([Xs[d] for d in Xs]),
                  np.concatenate([Y[d] for d in Y]),
                  batch_size=batch_size,
                  epochs=trainIters)

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
    parser.add_argument("--supervisedSegmenter", action='store_true')
    parser.add_argument("--supervisedAE", action='store_true')
    parser.add_argument("--debug", action='store_true')
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
    wordHidden = checkpoint.get('wordHidden', int(args.wordHidden) if args.wordHidden else 100 if ACOUSTIC else 40)
    uttHidden = checkpoint.get('uttHidden', int(args.uttHidden) if args.uttHidden else 500 if ACOUSTIC else 400)
    segHidden = checkpoint.get('segHidden', int(args.segHidden) if args.segHidden else 500 if ACOUSTIC else 100)
    wordDropout = checkpoint.get('wordDropout', float(args.wordDropout) if args.wordDropout else 0.25 if ACOUSTIC else 0.25)
    charDropout = checkpoint.get('charDropout', float(args.charDropout) if args.charDropout else 0.25 if ACOUSTIC else 0.25)
    maxChar = checkpoint.get('maxChar', int(args.maxChar) if args.maxChar else 500 if ACOUSTIC else 30)
    maxUtt = checkpoint.get('maxUtt', int(args.maxUtt) if args.maxUtt else 50 if ACOUSTIC else 10)
    maxLen = checkpoint.get('maxLen', int(args.maxLen) if args.maxLen else 100 if ACOUSTIC else 7)
    pretrainIters = checkpoint.get('pretrainIters', int(args.pretrainIters) if args.pretrainIters else 10 if ACOUSTIC else 10)
    trainNoSegIters = checkpoint.get('trainNoSegIters', int(args.trainNoSegIters) if args.trainNoSegIters else 10 if ACOUSTIC else 10)
    trainIters = checkpoint.get('trainIters', int(args.trainIters) if args.trainIters else 100 if ACOUSTIC else 80)
    METRIC = checkpoint.get('metric', args.metric if args.metric else 'mse' if ACOUSTIC else 'logprob')
    DEL_WT = checkpoint.get('delWt', float(args.delWt) if args.delWt else 1 if ACOUSTIC else 50)
    ONE_LETTER_WT = checkpoint.get('oneLetterWt', float(args.oneLetterWt) if args.oneLetterWt else 50 if ACOUSTIC else 10)
    SEG_WT = checkpoint.get('segWt', float(args.segWt) if args.segWt else 0 if ACOUSTIC else 0)
    N_SAMPLES = checkpoint.get('nSamples', int(args.nSamples) if args.nSamples else 50 if ACOUSTIC else 50)
    BATCH_SIZE = checkpoint.get('batchSize', int(args.batchSize) if args.batchSize else 16 if ACOUSTIC else 128)
    iteration = checkpoint.get('iteration', 0)
    pretrain = checkpoint.get('pretrain', True)
    DEBUG = args.debug
    wordDecLayers = 1
    RNN = recurrent.LSTM
    reverseUtt = True

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
    checkpoint['acoustic'] = ACOUSTIC
    checkpoint['metric'] = METRIC
    checkpoint['delWt'] = DEL_WT
    checkpoint['oneLetterWt'] = ONE_LETTER_WT
    checkpoint['segWt'] = SEG_WT
    checkpoint['nSamples'] = N_SAMPLES
    checkpoint['batchSize'] = BATCH_SIZE
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
    Xs = frameInputs2Utts(raw, vadBreaks, maxChar) if ACOUSTIC else texts2Xs(raw, maxChar, ctable)
    ## Xs_mask: mask of padding timesteps by utterance
    Xs_mask = getMasks(Xs)
    ## Xae: auto-encoder input (word-segmented input sequences by utterance)
    Xae = dict.fromkeys(doc_list)
    ## pSegs: segmentation proposal distribution
    pSegs = dict.fromkeys(doc_list)
    if ACOUSTIC:
        pSegs = segs2pSegsWithForced(segs_init, vadBreaks, alpha=0.05)
        pSegs = frameSegs2FrameSegsXUtt(pSegs, vadBreaks, maxChar)
    else:
        for doc in Xae:
            pSegs[doc] = .2 * np.ones((len(raw[doc]), maxChar))
            pSegs[doc][:, 0] = 1.
    ## Zero-out segmentation probability in padding regions
    for doc in pSegs:
        pSegs[doc][np.where(Xs_mask[doc])] = 0.
    ## segs: current segmentation proposal
    segs = dict.fromkeys(doc_list)
    ## Data loading finished, save checkpoint
    with open(logdir + '/checkpoint.obj', 'wb') as f:
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
        print('python scripts/autoencodeDecodeChars.py',
              '%s' % dataDir,
              '--acoustic' if ACOUSTIC else '',
              '--logfile %s' % logdir,
              '--metric %s' % METRIC,
              '--wordHidden %s' % wordHidden,
              '--uttHidden %s' % uttHidden,
              '--segHidden %s' % segHidden,
              '--wordDropout %s' % wordDropout,
              '--charDropout %s' % charDropout,
              '--maxLen %s' % maxLen,
              '--maxUtt %s' % maxUtt,
              '--maxChar %s' % maxChar,
              '--delWt %s' % DEL_WT,
              '--oneLetterWt %s' % ONE_LETTER_WT,
              '--segWt %s' % SEG_WT,
              '--nSample %s' % N_SAMPLES,
              '--batchSize %s' % BATCH_SIZE,
              '--pretrainIters %s' % pretrainIters,
              '--trainNoSegIters %s' % trainNoSegIters,
              '--trainIters %s' % trainIters,
              '--gpufrac %s' % args.gpufrac, file=f)

    print("Logging at", logdir)

    print("Constructing networks.")

    ## CONSTRUCT NETWORK GRAPHS
    ## 1. Auto-encoder
    ## a. Word-level AE
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

    ## c. Output masking
    if reverseUtt:
        premask = Lambda(lambda x: K.cast(K.any(K.reverse(inp, 1), axis=-1, keepdims=True), 'float32')*x)(resh2)
    else:
        premask = Lambda(lambda x: K.cast(K.any(inp, axis=-1, keepdims=True), 'float32')*x)(resh2)
    mask = Masking(mask_value=0.0)(premask)

    model = Model(input=inp, output=mask)
    if ACOUSTIC:
        model.compile(loss='mean_squared_error',
                      optimizer='adam')
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    model.summary()

    ## 2. Segmenter
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
    if load_models and os.path.exists(logdir + '/segmenter.h5'):
        print('Segmenter checkpoint found. Loading weights...')
        segmenter.load_weights(logdir + '/segmenter.h5', by_name=True)
    else:
        print('No segmenter checkpoint found. Keeping default initialization.')


    ## Unit testing of segmenter network by training on gold segmentations
    if args.supervisedAE or args.supervisedSegmenter:
        print('')
        print('Supervised network test (word-level segmentations)')

        ## Gold segmentations
        print('Using gold segmentations')
        if ACOUSTIC:
            _, goldseg = timeSegs2frameSegs(GOLDWRD)
            Y = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar)
        else:
            goldseg = gold
            Y = texts2Segs(gold, maxChar)

        if args.supervisedAE:
            trainAEOnly(model,
                        Xs,
                        Xs_mask,
                        Y,
                        trainIters,
                        BATCH_SIZE,
                        reverseUtt,
                        ACOUSTIC)

        if args.supervisedSegmenter:
            segsProposal = trainSegmenterOnly(segmenter,
                                              Xs,
                                              Xs_mask,
                                              Y[:,None],
                                              trainIters,
                                              BATCH_SIZE)
            print('Scoring segmentations')
            if ACOUSTIC:
                scores = getSegScores(gold['wrd'], frameSegs2timeSegs(intervals, segsProposal), acoustic=ACOUSTIC)
            else:
                scores = getSegScores(gold, segsProposal, acoustic=ACOUSTIC)
            printSegScores(scores, acoustic=ACOUSTIC)

        ## Random segmentations
        print('Using random segmentations')
        Y = sampleSegs(pSegs)
        if args.supervisedAE:
            trainAEOnly(model,
                        Xs,
                        Xs_mask,
                        Y,
                        trainIters,
                        BATCH_SIZE,
                        reverseUtt,
                        ACOUSTIC)

        if args.supervisedSegmenter:
            segsProposal = trainSegmenterOnly(segmenter,
                                              Xs,
                                              Xs_mask,
                                              Y[:,None],
                                              trainIters,
                                              BATCH_SIZE)
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
            Y = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar)

            if args.supervisedAE:
                trainAEOnly(model,
                            Xs,
                            Xs_mask,
                            Y,
                            trainIters,
                            reverseUtt,
                            ACOUSTIC)

            if args.supervisedSegmenter:
                segsProposal = trainSegmenterOnly(segmenter,
                                                  Xs,
                                                  Xs_mask,
                                                  Y[:,None],
                                                  trainIters)
                print('Scoring segmentations')
                scores = getSegScores(gold['phn'], frameSegs2timeSegs(intervals, segsProposal), acoustic=ACOUSTIC)
                printSegScores(scores, acoustic=ACOUSTIC)

            ## Random segmentations
            print('Using random segmentations')
            Y = sampleSegs(pSegs)
            if args.supervisedAE:
                trainAEOnly(model,
                            Xs,
                            Xs_mask,
                            Y,
                            trainIters,
                            reverseUtt,
                            ACOUSTIC)

            if args.supervisedSegmenter:
                segsProposal = trainSegmenterOnly(segmenter,
                                                  Xs,
                                                  Xs_mask,
                                                  Y[:,None],
                                                  trainIters)
                print('Scoring segmentations')
                scores = getSegScores(gold['phn'], frameSegs2timeSegs(intervals, segsProposal), acoustic=ACOUSTIC)
                printSegScores(scores, acoustic=ACOUSTIC)

        exit()





    ## Auto-encoder pretraining on initial segmentation proposal
    print()
    print("Pre-training networks.")
    while iteration < pretrainIters and pretrain:
        print('-' * 50)
        print('Iteration', iteration + 1)

        t0 = time.time()

        segs = sampleSegs(pSegs)
        Xae, deletedChars, oneLetter = XsSegs2Xae(Xs,
                                                  Xs_mask,
                                                  segs,
                                                  maxUtt,
                                                  maxLen)

        t1 = time.time()
        print('Auto-encoder input preprocessing completed in %.4fs.' %(t1-t0))

        Yae = dict.fromkeys(Xae)
        if reverseUtt:
            for doc in Yae:
                Yae[doc] = np.flip(Xae[doc], 1)
        else:
            for doc in Yae:
                Yae[doc] = Xae[doc]

        print("Total deleted chars: %d" %(sum([deletedChars[d].sum() for d in deletedChars])))
        print("Total one-letter words: %d" %(sum([oneLetter[d].sum() for d in oneLetter])))

        if iteration == 0:
            print('Training segmenter network on random segmentation.')
            segmenter.fit(np.concatenate([Xs[d] for d in Xs]), np.concatenate([segs[d] for d in segs]))
        print('Training auto-encoder network on random segmentation.')
        model.fit(np.concatenate([Xae[d] for d in Xae]), np.concatenate([Yae[d] for d in Yae]), batch_size=BATCH_SIZE, epochs=1)

        # Correctness checks for NN masking
        if DEBUG:
            for doc in Xae:
                out = model.predict(Xae[doc], batch_size=BATCH_SIZE)
                print('Document: %s' %doc)
                print('Timesteps in input: %s' %Xae[doc].any(-1).sum())
                print('Timesteps in output: %s (should equal timesteps in input)' %out.any(-1).sum())
                print('Deleted timesteps: %s' %int(deletedChars[doc].sum()))
                print('Timesteps + deleted: %s (should be %s)' % (out.any(-1).sum() + int(deletedChars[doc].sum()), raw_cts[doc]))
                print('')
            
        model.save(logdir + '/model.h5')
        with open(logdir + '/checkpoint.obj', 'wb') as f:
            checkpoint['iteration'] = iteration
            pickle.dump(checkpoint, f)

        printSome = not ACOUSTIC
        if printSome:
            toPrint = 10
            for doc in Xae:
                preds = model.predict(Xae[doc][:toPrint], batch_size=BATCH_SIZE)
                print(preds)
                print('Sum of autoencoder predictions: %s' %preds.sum())
                if reverseUtt:
                    preds = np.flip(preds, 1)

                for utt in range(toPrint):
                    thisSeg = segs[doc][utt]
                    rText = reconstructUtt(raw[doc][utt], thisSeg, maxUtt)

                    print('Target: %s' %realize(rText, maxLen, maxUtt))
                    print('Reconstruction:', end=' ')
                    for wi in range(maxUtt):
                        guess = ctable.decode(preds[utt, wi])
                        print(guess, end=" ")
                    print("\n")

        iteration += 1
        if iteration == pretrainIters:
            pretrain = False
            iteration = 0

    ## Pretraining finished, update checkpoint
    with open(logdir + '/checkpoint.obj', 'wb') as f:
        checkpoint['pretrain'] = pretrain
        checkpoint['iteration'] = iteration
        pickle.dump(checkpoint, f)

    if segsProposal == None:
        segsProposal = dict.fromkeys(doc_list)
        for doc in doc_list:
            if ACOUSTIC:
                segsProposal[doc] = np.zeros(segs_init[doc].shape)
            else:
                segsProposal[doc] = np.zeros((Xs[doc].shape[0], maxChar))
    if ACOUSTIC:
        segs = segs_init
    segScores = {'wrd': dict.fromkeys(doc_list), 'phn': dict.fromkeys(doc_list)}
    segScores['wrd']['##overall##'] = [(0,0,0), None, (0,0,0), None]
    segScores['phn']['##overall##'] = [(0,0,0), None, (0,0,0), None]
    segSamples = dict.fromkeys(doc_list)
    segProbs = dict.fromkeys(doc_list)
    scores = dict.fromkeys(doc_list)
    for doc in doc_list:
        segSamples[doc] = np.zeros((len(Xs[doc]), N_SAMPLES, maxChar))
        segProbs[doc] = np.zeros((len(Xs[doc]), maxChar))
        scores[doc] = np.zeros((len(Xs[doc]), N_SAMPLES, maxUtt))
    print()

    print('Training networks jointly.')
    while iteration < trainIters:
        epochLoss = 0
        epochDel = 0
        epochOneL = 0
        epochSeg = 0

        print()
        print('-' * 50)
        print('Iteration', iteration)

        if iteration <= trainNoSegIters:
            print('Using initialization as proposal distribution.')
        else:
            print('Using segmenter network as proposal distribution.')
            for doc in pSegs:
                pSegs = segmenter.predict(Xs[doc], batch_size = BATCH_SIZE)
                pSegs = .9 * pSegs + .1 * .5 * np.ones(pSegs.shape)
                if ACOUSTIC:
                    pSegs[np.where(vadBreaks[doc])] = 1.
                else:
                    pSegs[:,0] = 1.
            ## Zero-out segmentation probability in padding regions
            for doc in pSegs:
                pSegs[doc][np.where(Xs_mask[doc])] = 0.

        ts0 = time.time()
        print('')
        for s in range(N_SAMPLES):
            sys.stdout.write('\rSample %d/%d' %(s+1, N_SAMPLES))
            sys.stdout.flush()
            segs = sampleSegs(pSegs)
            for doc in segSamples:
                segSamples[doc][:, s, :] = segs[doc]

            Xae, deletedChars, oneLetter = XsSegs2Xae(Xs,
                                                      Xs_mask,
                                                      segs,
                                                      maxUtt,
                                                      maxLen)

            if reverseUtt:
                for doc in doc_list:
                    Yae[doc] = np.flip(Xae[doc], 1)
            else:
                Yae = Xae

            for doc in scores:
                scores[doc][:,s,:] = lossByUtt(model, Xae[doc], Yae[doc], BATCH_SIZE, metric = METRIC)
                scores[doc][:,s,:] -= deletedChars[doc] * DEL_WT
                scores[doc][:,s,:] -= (oneLetter[doc] / maxUtt)[:, None] * ONE_LETTER_WT
                scores[doc][:,s,:] -= (segs[doc].sum(-1, keepdims=True) / maxUtt) * SEG_WT
        ts1 = time.time()

        for doc in segSamples:
            segProbs[doc], segsProposal[doc] = guessSegTargets(scores[doc],
                                                               segSamples[doc],
                                                               pSegs[doc],
                                                               'importance')

        print('Sampling completed in %ds.' %(ts1-ts0))

        Xae, deletedChars, oneLetter = XsSegs2Xae(Xs,
                                                  Xs_mask,
                                                  segsProposal,
                                                  maxUtt,
                                                  maxLen)

        if reverseUtt:
            for doc in doc_list:
                Yae[doc] = np.flip(Xae[doc], 1)
        else:
            Yae = Xae

        print('Updating auto-encoder network')
        h = model.fit(np.concatenate([Xae[d] for d in Xae]),
                         np.concatenate([Yae[d] for d in Yae]),
                         batch_size=BATCH_SIZE,
                         epochs=1)

        print('Updating segmenter network')
        segmenter.fit(np.concatenate([Xs[d] for d in Xs]),
                      np.concatenate([segProbs[d] for d in segProbs]),
                      batch_size=BATCH_SIZE,
                      epochs=1)

        epochLoss = h.history['loss'][0]
        epochDel += int(sum([deletedChars[d].sum() for d in deletedChars]))
        epochOneL += int(sum([oneLetter[d].sum() for d in oneLetter]))
        epochSeg += int(sum([segsProposal[d].sum() for d in segsProposal]))

        model.save(logdir + '/model.h5')
        segmenter.save(logdir + '/segmenter.h5')
        with open(logdir + '/checkpoint.obj', 'wb') as f:
            checkpoint['iteration'] = iteration
            checkpoint['segsProposal'] = segsProposal
            checkpoint['deletedChars'] = deletedChars
            checkpoint['oneLetter'] = oneLetter
            pickle.dump(checkpoint, f)

        print('Total frames:', raw_total)
        print("Loss:", epochLoss)
        print("Deletions:", epochDel)
        print("One letter words:", epochOneL)
        print("Total segmentation points:", epochSeg)
        segScore = writeLog(iteration, epochLoss, epochDel, epochOneL, epochSeg,
                            gold, segsProposal, logdir, intervals if ACOUSTIC else None, ACOUSTIC, print_headers=iteration == 0)
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
            writeTimeSegs(frameSegs2timeSegs(intervals, segsProposal), out_file=logdir, TextGrid=False)
            writeTimeSegs(frameSegs2timeSegs(intervals, segsProposal), out_file=logdir, TextGrid=True)
        else:
            printSegScores(getSegScores(gold, segsProposal, ACOUSTIC), ACOUSTIC)
            writeSolutions(logdir, model, segmenter,
                           segsProposal[doc_list[0]], gold[doc_list[0]], iteration)

        doc_ix = 0
        iteration += 1

    print("Logs in", logdir)

