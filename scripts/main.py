from __future__ import print_function, division
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Model, Sequential, load_model
try:
    from keras.engine.training import slice_X
except:
    from keras.engine.training import _slice_arrays as slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Input, Reshape, Merge, merge, Lambda, Dropout, Masking, multiply, Conv1D
from keras import backend as K
from keras import optimizers
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
import signal
from itertools import izip
from collections import defaultdict
from echo_words import CharacterTable, pad
from capacityStatistics import getPseudowords
from ae_io import *
from data_handling import *
from sampling import *
from scoring import *
from network import *
from plotting import *

class sigHandler(object):
    def __init__(self):
        self.sigint_received = False
        self.interrupt_allowed = True

    def disallow_interrupt(self):
        self.interrupt_allowed = False

    def allow_interrupt(self):
        self.interrupt_allowed = True
        if self.sigint_received == True:
            sys.exit(0)

    def interrupt(self, signum = None, frame = None):
        self.sigint_received = True
        if self.interrupt_allowed == True:
            sys.exit(0)

sig = sigHandler()
signal.signal(signal.SIGINT, sig.interrupt)

argmax = lambda array: max(izip(array, xrange(len(array))))[1]
argmin = lambda array: min(izip(array, xrange(len(array))))[1]

adam = optimizers.adam(clipnorm=1.)
nadam = optimizers.Nadam(clipnorm=1.)
rmsprop = optimizers.RMSprop(clipnorm=1.)
optim_map = {'adam': adam, 'nadam': nadam, 'rmsprop': rmsprop}

if __name__ == "__main__":

    ##################################################################################
    ##################################################################################
    ##
    ##  Process CLI args
    ##
    ##################################################################################
    ##################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("dataDir")
    parser.add_argument("--acoustic", action='store_true')
    parser.add_argument("--noSegNet", action='store_true')
    parser.add_argument("--noAENet", action='store_true')
    parser.add_argument("--supervisedSegmenter", action='store_true')
    parser.add_argument("--supervisedAE", action='store_true')
    parser.add_argument("--supervisedAEPhon", action='store_true')
    parser.add_argument("--reverseUtt", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--nResample")
    parser.add_argument("--crossValDir")
    parser.add_argument("--evalFreq")
    parser.add_argument("--algorithm")
    parser.add_argument("--optimizer")
    parser.add_argument("--wordHidden")
    parser.add_argument("--uttHidden")
    parser.add_argument("--segHidden")
    parser.add_argument("--wordDropout")
    parser.add_argument("--charDropout")
    parser.add_argument("--depth")
    parser.add_argument("--segShift")
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
    parser.add_argument("--initialSegProb")
    parser.add_argument("--interpolationRate")
    parser.add_argument("--logfile")
    parser.add_argument("--gpufrac")
    parser.add_argument("--nViz")
    args = parser.parse_args()
    try:
        args.gpufrac = float(args.gpufrac)
    except:
        args.gpufrac = None





    ##################################################################################
    ##################################################################################
    ##
    ## Set up GPU params
    ##
    ##################################################################################
    ##################################################################################

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpufrac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    usingGPU = is_gpu_available()
    print('Using GPU: %s' %usingGPU)
    K.set_session(sess)
    K.set_learning_phase(1)





    ##################################################################################
    ##################################################################################
    ##
    ## Load any saved data/parameters
    ##
    ##################################################################################
    ##################################################################################

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
        if not (args.supervisedAE or args.supervisedAEPhon or args.supervisedSegmenter):
            load_models = True

    if load_models and os.path.exists(logdir + '/checkpoint.obj'):
        print('Training checkpoint data found. Loading...')
        with open(logdir + '/checkpoint.obj', 'rb') as f:
            checkpoint = pickle.load(f)
    else:
        print('No training checkpoint found. Starting training from beginning.')
        checkpoint = {}





    ##################################################################################
    ##################################################################################
    ##
    ##  Initialize system parameters
    ##
    ##################################################################################
    ##################################################################################

    dataDir = args.dataDir
    ACOUSTIC = checkpoint.get('acoustic', args.acoustic)
    SEG_NET = checkpoint.get('segNet', not args.noSegNet)
    AE_NET = checkpoint.get('AENet', not args.noAENet)
    assert SEG_NET or AE_NET, 'At least one of the networks must be turn on.'
    ALGORITHM = args.algorithm if args.algorithm else checkpoint.get('algorithm', 'importance')
    assert ALGORITHM in ['importance', '1best', 'viterbi'], 'Algorithm "%s" is not supported. Use one of: %s' %(ALGORITHM, ', '.join['importance"', '1best', 'viterbi'])
    assert AE_NET or not ALGORITHM == 'viterbi', 'Viterbi sampling requires the AE network to be turned on.'
    OPTIM = args.optimizer if args.optimizer else checkpoint.get('optimizer', 'nadam' if ACOUSTIC else 'nadam')
    REVERSE_UTT = args.reverseUtt
    N_RESAMPLE = checkpoint.get('nResample', int(args.nResample) if args.nResample else None if ACOUSTIC else None)
    assert ACOUSTIC or not N_RESAMPLE, 'Resampling disallowed in character mode since it does not make any sense'
    crossValDir = args.crossValDir if args.crossValDir else checkpoint.get('crossValDir', None)
    EVAL_FREQ = int(args.evalFreq) if args.evalFreq else checkpoint.get('evalFreq', 50)
    MASK_VALUE = 0 if ACOUSTIC else 1
    wordHidden = checkpoint.get('wordHidden', int(args.wordHidden) if args.wordHidden else 100 if ACOUSTIC else 80)
    uttHidden = checkpoint.get('uttHidden', int(args.uttHidden) if args.uttHidden else 500 if ACOUSTIC else 400)
    segHidden = checkpoint.get('segHidden', int(args.segHidden) if args.segHidden else 500 if ACOUSTIC else 100)
    wordDropout = checkpoint.get('wordDropout', float(args.wordDropout) if args.wordDropout else 0.25 if ACOUSTIC else 0.25)
    charDropout = checkpoint.get('charDropout', float(args.charDropout) if args.charDropout else 0.25 if ACOUSTIC else 0.5)
    maxChar = checkpoint.get('maxChar', int(args.maxChar) if args.maxChar else 500 if ACOUSTIC else 30)
    maxUtt = checkpoint.get('maxUtt', int(args.maxUtt) if args.maxUtt else 50 if ACOUSTIC else 10)
    maxLen = checkpoint.get('maxLen', int(args.maxLen) if args.maxLen else 100 if ACOUSTIC else 7)
    DEPTH = checkpoint.get('depth', int(args.depth) if args.depth != None else 1)
    SEG_SHIFT = int(args.segShift) if args.segShift != None else checkpoint.get('segShift', 0 if ACOUSTIC else 0)
    pretrainIters = int(args.pretrainIters) if args.pretrainIters else checkpoint.get('pretrainIters', 10 if ACOUSTIC else 10)
    trainNoSegIters = int(args.trainNoSegIters) if args.trainNoSegIters else checkpoint.get('trainNoSegIters', 10 if ACOUSTIC else 10)
    trainIters = int(args.trainIters) if args.trainIters else checkpoint.get('trainIters', 100 if ACOUSTIC else 80)
    METRIC = args.metric if args.metric else checkpoint.get('metric', 'mse' if ACOUSTIC else 'logprob')
    DEL_WT = float(args.delWt) if args.delWt else checkpoint.get('delWt', 50 if ACOUSTIC else 50)
    ONE_LETTER_WT = float(args.oneLetterWt) if args.oneLetterWt else checkpoint.get('oneLetterWt', 50 if ACOUSTIC else 10)
    SEG_WT = float(args.segWt) if args.segWt else checkpoint.get('segWt', 0 if ACOUSTIC else 0)
    N_SAMPLES = int(args.nSamples) if args.nSamples else checkpoint.get('nSamples', 100 if ACOUSTIC else 100)
    BATCH_SIZE = int(args.batchSize) if args.batchSize else checkpoint.get('batchSize', 128 if ACOUSTIC else 128)
    SAMPLING_BATCH_SIZE = int(args.samplingBatchSize) if args.samplingBatchSize else checkpoint.get('samplingBatchSize', 128 if ACOUSTIC else 128)
    INITIAL_SEG_PROB = float(args.initialSegProb) if args.initialSegProb else checkpoint.get('initialSegProb', 0.2 if ACOUSTIC else 0.2)
    assert INITIAL_SEG_PROB >= 0 and INITIAL_SEG_PROB <= 1, 'Invalid value for initialSegProb (%.2f) -- must be between 0 and 1' %INITIAL_SEG_PROB
    INTERPOLATION_RATE = float(args.interpolationRate) if args.interpolationRate else checkpoint.get('interpolationRate', 0.1 if ACOUSTIC else 0.1)
    assert INTERPOLATION_RATE >= 0 and INTERPOLATION_RATE <= 1, 'Invalid value for interpolationRate (%.2f) -- must be between 0 and 1' %INTERPOLATION_RATE
    iteration = checkpoint.get('iteration', 0)
    pretrain = checkpoint.get('pretrain', True)
    DEBUG = args.debug
    RNN = recurrent.LSTM
    N_VIZ = checkpoint.get('nViz', int(args.nViz) if args.nViz != None else 10)
    if SEG_NET and not AE_NET:
        METRIC = 'logprobbinary'




    ##################################################################################
    ##################################################################################
    ##
    ##  Save system parameters to checkpoint object for training resumption
    ##
    ##################################################################################
    ##################################################################################

    checkpoint['acoustic'] = ACOUSTIC
    checkpoint['segNet'] = SEG_NET
    checkpoint['AENet'] = AE_NET
    checkpoint['algorithm'] = ALGORITHM
    checkpoint['optimizer'] = OPTIM
    checkpoint['reverseUtt'] = REVERSE_UTT
    checkpoint['nResample'] = N_RESAMPLE
    checkpoint['crossValDir'] = crossValDir
    checkpoint['evalFreq'] = EVAL_FREQ
    checkpoint['wordHidden'] = wordHidden
    checkpoint['uttHidden'] = uttHidden
    checkpoint['segHidden'] = segHidden
    checkpoint['wordDropout'] = wordDropout
    checkpoint['charDropout'] = charDropout
    checkpoint['maxChar'] = maxChar
    checkpoint['maxUtt'] = maxUtt
    checkpoint['maxLen'] = maxLen
    checkpoint['depth'] = DEPTH
    checkpoint['segShift'] = SEG_SHIFT
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
    checkpoint['initialSegProb'] = INITIAL_SEG_PROB
    checkpoint['interpolationRate'] = INTERPOLATION_RATE
    checkpoint['iteration'] = iteration
    checkpoint['pretrain'] = pretrain
    checkpoint['evalFreq'] = EVAL_FREQ
    checkpoint['nViz'] = N_VIZ





    ##################################################################################
    ##################################################################################
    ##
    ##  Load training data
    ##
    ##################################################################################
    ##################################################################################

    print()
    print('Pre-processing training data')
    doc_indices, doc_list, charDim, raw_total, Xs, Xs_mask, gold, other = processInputDir(dataDir, checkpoint, maxChar, ACOUSTIC)
    if ACOUSTIC:
        intervals, vad, vadBreaks, SEGFILE, GOLDWRD, GOLDPHN = other
    else:
        ctable = other

    if crossValDir:
        print()
        print('Pre-processing cross-validation data')
        doc_indices_cv, doc_list_cv, charDim_cv, raw_total_cv, Xs_cv, Xs_mask_cv, gold_cv, other_cv = processInputDir(dataDir, checkpoint, maxChar, ACOUSTIC)
        assert charDim == charDim_cv, 'ERROR: Training and cross-validation data have different dimensionality (%i, %i)' %(charDim, charDim_cv)
        if ACOUSTIC:
            intervals_cv, vad_cv, vadBreaks_cv, SEGFILE_CV, GOLDWRD_CV, GOLDPHN_CV = other_cv
        else:
            ctable_cv = other

    ## Randomly select utterances for visualization
    utt_ids = checkpoint.get('uttIDs', np.random.choice(len(Xs), size=N_VIZ, replace=False))
    if crossValDir:
        utt_ids_cv = checkpoint.get('uttIDsCV', np.random.choice(len(Xs_cv), size=N_VIZ, replace=False))
    checkpoint['uttIDs'] = utt_ids
    if crossValDir:
        checkpoint['uttIDsCV'] = utt_ids_cv

    ## pSegs: segmentation proposal distribution
    if ACOUSTIC:
        pSegs = INITIAL_SEG_PROB * np.ones((len(Xs), maxChar, 1))
        pSegs[np.where(vad)] = 1.
    else:
        pSegs = INITIAL_SEG_PROB * np.ones((len(Xs), maxChar, 1))
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





    ##################################################################################
    ##################################################################################
    ##
    ##  Save system parameters to log file
    ##
    ##################################################################################
    ##################################################################################

    with open(logdir + '/params.txt', 'wb') as f:
        print('Model parameters:', file=f)
        if ACOUSTIC:
            print('  Input type: Acoustic', file=f)
            print('  Training data location: %s' % dataDir, file=f)
            if SEGFILE:
                print('  Training initial segmentation file: %s' % SEGFILE, file=f)
            if GOLDWRD:
                print('  Training gold word segmentation file: %s' % GOLDWRD, file=f)
            if GOLDPHN:
                print('  Training gold phoneme segmentation file: %s' % GOLDPHN, file=f)
        else:
            print('  Input type: Text', file=f)
        if crossValDir:
            print('  Cross-validation data location: %s' % crossValDir, file=f)
            if ACOUSTIC:
                if SEGFILE:
                    print('  Cross-validation segmentation file: %s' % SEGFILE_CV, file=f)
                if GOLDWRD:
                    print('  Cross-validation gold word segmentation file: %s' % GOLDWRD_CV, file=f)
                if GOLDPHN:
                    print('  Cross-validation gold phone segmentation file: %s' % GOLDPHN_CV, file=f)
        print('  Using segmenter network: %s' % SEG_NET, file=f)
        print('  Using auto-encoder network: %s' % AE_NET, file=f)
        print('  Unit testing segmenter network: %s' % args.supervisedSegmenter, file=f)
        print('  Unit testing auto-encoder network: %s' % args.supervisedAE, file=f)
        print('  Unit testing phonological auto-encoder network: %s' % args.supervisedAEPhon, file=f)
        print('  Reversing word order in reconstruction targets: %s' % REVERSE_UTT, file=f)
        if N_RESAMPLE:
            print('  Resampling discovered words to maximum word length: %s' % N_RESAMPLE, file=f)
        print('  Search algorithm: %s' % ALGORITHM, file=f)
        print('  Optimizer: %s' % OPTIM, file=f)
        print('  Autoencoder loss function: %s' % METRIC, file=f)
        print('  Word layer hidden units: %s' % wordHidden, file=f)
        print('  Utterance layer hidden units: %s' % uttHidden, file=f)
        print('  Segmenter network hidden units: %s' % segHidden, file=f)
        print('  Word dropout rate: %s' % wordDropout, file=f)
        print('  Character dropout rate: %s' % charDropout, file=f)
        print('  RNN depth: %s' % DEPTH, file=f)
        print('  Segmenter target offset: %s' % SEG_SHIFT, file=f)
        print('  Loss metric: %s' % METRIC, file=f)
        print('  Pretraining iterations: %s' % pretrainIters, file=f)
        print('  Training iterations without segmenter network: %s' % trainNoSegIters, file=f)
        print('  Training iterations (total): %s' % trainIters, file=f)
        print('  Maximum utterance length (characters): %s' % maxChar, file=f)
        print('  Maximum utterance length (words): %s' % maxUtt, file=f)
        print('  Maximum word length (characters): %s' % maxLen, file=f)
        print('  Deletion penalty: %s' % DEL_WT, file=f)
        print('  One letter segment penalty: %s' % ONE_LETTER_WT, file=f)
        print('  Segmentation penalty: %s' % SEG_WT, file=f)
        print('  Number of samples per batch: %s' % N_SAMPLES, file=f)
        print('  Batch size (network training): %s' % BATCH_SIZE, file=f)
        print('  Batch size (sampling): %s' % SAMPLING_BATCH_SIZE, file=f)
        print('  Initial segmentation probability: %s' % INITIAL_SEG_PROB, file=f)
        print('  Rate of interpolation of segmenter distribution with uniform: %s' % INTERPOLATION_RATE, file=f)
        print('  Logging directory path: %s' % logdir, file=f)
        print('  Number of examples to visualize: %s' % N_VIZ, file=f)
        print('  Maximum allocatable fraction of GPU: %s' % args.gpufrac, file=f)
        print('', file=f)
        print('Command line call to repro/resume:', file=f)
        print('', file=f)
        print('python scripts/main.py',
              '%s' % dataDir,
              '--acoustic' if ACOUSTIC else '',
              '--noSegNet' if not SEG_NET else '',
              '--noAENet' if not AE_NET else '',
              '--supervisedSegmenter' if args.supervisedSegmenter else '',
              '--supervisedAE' if args.supervisedAE else '',
              '--supervisedAEPhon' if args.supervisedAEPhon else '',
              '--reverseUtt' if REVERSE_UTT else '',
              '--nResample %s' % N_RESAMPLE if N_RESAMPLE else '',
              '--crossValDir %s' % crossValDir if crossValDir else '',
              '--evalFreq %s' % EVAL_FREQ,
              '--algorithm %s' % ALGORITHM,
              '--optimizer %s' % OPTIM,
              '--wordHidden %s' % wordHidden,
              '--uttHidden %s' % uttHidden,
              '--segHidden %s' % segHidden,
              '--wordDropout %s' % wordDropout,
              '--charDropout %s' % charDropout,
              '--depth %s' % DEPTH,
              '--segShift %s' % SEG_SHIFT,
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
              '--initialSegProb %s' % INITIAL_SEG_PROB,
              '--interpolationRate %s' % INTERPOLATION_RATE,
              '--logfile %s' % logdir[5:],
              '--nViz %s' % N_VIZ,
              '--gpufrac %s' % args.gpufrac, file=f)

    print("Logging at", logdir)




    ##################################################################################
    ##################################################################################
    ##
    ##  Construct network graphs
    ##
    ##################################################################################
    ##################################################################################

    print("Constructing networks.")

    networkType = 'AESeg'  ## Only option for now... may implement other architectures later
    if networkType == 'AESeg':
        if AE_NET:
            ## AUTO-ENCODER NETWORK

            wLen = N_RESAMPLE if N_RESAMPLE else maxLen

            ## INPUT
            inp = Input(shape=(maxUtt, wLen, charDim), name='AEfullInput')

            ## WORD ENCODER
            wordEncoder = Sequential(name='wordEncoder')
            wordEncoder.add(Dropout(charDropout,input_shape=(wLen, charDim), noise_shape=(1, wLen, 1)))
            #wordEncoder.add(Conv1D(wordHidden, 10))
            wordEncoder.add(RNN(wordHidden))

            ## WORDS ENCODER
            wordsEncoded = TimeDistributed(wordEncoder, name='wordsEncoded')(inp)
            wordsEncoded = TimeDistributed(Dropout(wordDropout, noise_shape=(1,)), name='wordDropout')(wordsEncoded)

            ## UTTERANCE ENCODER
            uttEncoder = Sequential(name="uttEncoder")
            uttEncoder.add(RNN(uttHidden, return_sequences=False, input_shape=(maxUtt, wordHidden)))
            uttEncoded = uttEncoder(wordsEncoded)

            ## UTTERANCE DECODER
            uttDecoder = Sequential(name='uttDecoder')
            uttDecoder.add(RepeatVector(maxUtt, input_shape=(uttHidden,)))
            uttDecoder.add(RNN(uttHidden, return_sequences=True))
            uttDecoder.add(TimeDistributed(Dense(wordHidden)))
            uttDecoder.add(Activation('linear'))
            uttDecoded = uttDecoder(uttEncoded)

            ## WORD DECODER
            wordDecoder = Sequential(name='wordDecoder')
            wordDecoder.add(RepeatVector(wLen, input_shape=(wordHidden,)))
            wordDecoder.add(RNN(wordHidden, return_sequences=True))
            wordDecoder.add(TimeDistributed(Dense(charDim)))
            wordDecoder.add(Activation('linear' if ACOUSTIC else 'softmax'))

            ## OUTPUT LAYER
            wordsDecoded = TimeDistributed(wordDecoder, name='wordsDecoded')(uttDecoded)

            ## OUTPUT MASKIONG
            if REVERSE_UTT:
                mask = Lambda(lambda x: x * K.cast(K.any(K.reverse(inp, (1,2)), -1, keepdims=True), 'float32'),
                              name='AEfull-output-premask')(wordsDecoded)
            else:
                mask = Lambda(lambda x: x * K.cast(K.any(inp, -1, keepdims=True), 'float32'),
                              name='AEfull-output-premask')(wordsDecoded)
            wordsDecoded = Masking(mask_value=0, name='AEfull-mask')(mask)

            ## PHONOLOGICAL AE
            inp_phon = Input(shape=(wLen, charDim), name='AEphonInput')
            wordDecoded = wordDecoder(wordEncoder(inp_phon))
            if REVERSE_UTT:
                mask = Lambda(lambda x: x * K.cast(K.any(K.reverse(inp_phon, (1,2)), -1, keepdims=True), 'float32'),
                              name='AEphon-output-premask')(wordDecoded)
            else:
                mask = Lambda(lambda x: x * K.cast(K.any(inp_phon, -1, keepdims=True), 'float32'),
                              name='AEphon-output-premask')(wordDecoded)
            wordDecoded = Masking(mask_value=0, name='AEphon-mask')(mask)
            ae_phon = Model(input=inp_phon, output=wordDecoded)
            if ACOUSTIC:
                ae_phon.compile(loss="mean_squared_error",
                                optimizer=optim_map[OPTIM])
            else:
                ae_phon.compile(loss=masked_categorical_crossentropy,
                                optimizer=optim_map[OPTIM],
                                metrics=[masked_categorical_accuracy])
            print('Phonological auto-encoder network summary')
            ae_phon.summary()
            print()

            ## UTTERANCE AE
            inp_utt = Input(shape=(maxUtt, wordHidden), name='AEuttInput')
            decoderUtt = uttEncoder(inp_utt)
            decoderUtt = uttDecoder(decoderUtt)
            ae_utt = Model(input=inp_utt, output=decoderUtt)
            ae_utt.compile(loss="mean_squared_error",
                           optimizer=optim_map[OPTIM])

            print('Utterance auto-encoder network summary')
            ae_utt.summary()
            print()

            ## WORD EMBEDDING MODEL
            embed_word = Model(input=inp_phon, output=wordEncoder(inp_phon))
            embed_word.compile(loss="mean_squared_error",
                               optimizer=optim_map[OPTIM])
            print('Word embedding network summary')
            embed_word.summary()
            print()

            ## WORD EMBEDDINGS MODEL
            embed_words = Model(input=inp, output=wordsEncoded)
            embed_words.compile(loss="mean_squared_error",
                                optimizer=optim_map[OPTIM])
            print('Word embeddings network summary')
            embed_words.summary()
            print()

            ## AE MODEL CREATION
            # model = Model(input=inp, outputs=[wordsDecoder, utt2Words])
            ae_full = Model(input=inp, output=wordsDecoded)

            ## AE MODEL COMPILATION
            if ACOUSTIC:
                ae_full.compile(loss="mean_squared_error",
                                optimizer=optim_map[OPTIM])
            else:
                # model.compile(loss=[masked_categorical_crossentropy, 'mean_squared_error'],
                #               optimizer=optim_map[OPTIM],
                #               metrics=[masked_categorical_accuracy, 'mean_squared_error'])
                ae_full.compile(loss=masked_categorical_crossentropy,
                                optimizer=optim_map[OPTIM],
                                metrics=[masked_categorical_accuracy])

            ## PRINT AE MODEL
            print('Auto-encoder network summary')
            ae_full.summary()
            print()

            ## SAVE AE MODEL
            ae_full.save(logdir + '/model_init.h5')


        if SEG_NET:
            ## SEGMENTER NETWORK
            segInput = Input(shape=(maxChar+SEG_SHIFT, charDim), name='SegmenterInput')
            segMaskInput = Input(shape=(maxChar+SEG_SHIFT,1), name='SegmenterMaskInput')

            segmenter = Sequential(name="Segmenter")
            segmenter.add(RNN(segHidden, return_sequences=True, input_shape=(maxChar+SEG_SHIFT, charDim)))
            segmenter.add(TimeDistributed(Dense(1)))
            segmenter.add(Activation("sigmoid"))
            segmenterPremask = Lambda(lambda x: x[0] * (1- K.cast(x[1], 'float32')), name='Seg-output-premask')([segmenter(segInput), segMaskInput])
            segmenter = Masking(mask_value=0.0, name='Seg-mask')(segmenterPremask)
            segmenter = Model(inputs=[segInput, segMaskInput], output = segmenter)
            segmenter.compile(loss="binary_crossentropy",
                              optimizer=optim_map[OPTIM])
            print('Segmenter network summary')
            segmenter.summary()
            print('')

            segmenter.save(logdir + '/segmenter_init.h5')

    if AE_NET:
        if load_models and os.path.exists(logdir + '/model.h5'):
            print('Autoencoder checkpoint found. Loading weights...')
            ae_full.load_weights(logdir + '/model.h5', by_name=True)
        else:
            print('No autoencoder checkpoint found. Keeping default initialization.')
            ae_full.save(logdir + '/model.h5')
    if SEG_NET:
        if load_models and os.path.exists(logdir + '/segmenter.h5'):
            print('Segmenter checkpoint found. Loading weights...')
            segmenter.load_weights(logdir + '/segmenter.h5', by_name=True)
        else:
            print('No segmenter checkpoint found. Keeping default initialization.')
            segmenter.save(logdir + '/segmenter.h5')






    ##################################################################################
    ##################################################################################
    ##
    ##  Unit test networks by training on fixed segmentations
    ##  (Skipped unless --supervisedAE or --supervisedSegmenter are used)
    ##
    ##################################################################################
    ##################################################################################

    if (args.supervisedAE and AE_NET) or (args.supervisedAEPhon and AE_NET) or (args.supervisedSegmenter and SEG_NET):
        print('')
        print('Network unit testing')
        segsProposalXDoc = dict.fromkeys(doc_list)

        ## Phonological auto-encoder test
        if not ACOUSTIC or GOLDWRD:
            if ACOUSTIC:
                _, goldseg = timeSegs2frameSegs(GOLDWRD)
                Y = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
            else:
                goldseg = gold
                Y = texts2Segs(gold, maxChar)
            if args.supervisedAEPhon:
                print('Unit testing phonological auto-encoder network on gold segmentation')
                ## Re-initialize network weights in case any training has already happened
                ae_full.load_weights(logdir + '/model_init.h5', by_name=True)

                for i in range(trainIters):
                    print('Iteration %d' % (i + 1))
                    Xae = trainAEPhonOnly(ae_phon,
                                          Xs,
                                          Xs_mask,
                                          Y,
                                          maxLen,
                                          1,
                                          BATCH_SIZE,
                                          REVERSE_UTT,
                                          N_RESAMPLE)

                    if REVERSE_UTT:
                        Yae = np.flip(Xae, 1)
                    else:
                        Yae = Xae

                    plotPredsWrd(utt_ids,
                                 ae_phon,
                                 Xae,
                                 Yae,
                                 logdir,
                                 'rand',
                                 i,
                                 BATCH_SIZE,
                                 DEBUG)

        ## Random segmentations
        Y = sampleSeg(pSegs)
        if args.supervisedAE:
            print('Unit testing auto-encoder network on fixed random segmentation')
            ## Re-initialize network weights in case any training has already happened
            ae_full.load_weights(logdir + '/model_init.h5', by_name=True)

            for i in range(trainIters):
                print('Iteration %d' % (i + 1))
                Xae = trainAEOnly(ae_full,
                                  ae_phon,
                                  ae_utt,
                                  embed_words,
                                  Xs,
                                  Xs_mask,
                                  Y,
                                  maxUtt,
                                  maxLen,
                                  1,
                                  BATCH_SIZE,
                                  REVERSE_UTT,
                                  N_RESAMPLE)

                plotPredsUtt(utt_ids,
                             ae_full,
                             Xae,
                             getYae(Xae, REVERSE_UTT),
                             logdir,
                             'rand',
                             i+1,
                             BATCH_SIZE,
                             DEBUG)

                if not ACOUSTIC:
                    printReconstruction(10, ae_full, Xae, ctable, BATCH_SIZE, REVERSE_UTT)

        if args.supervisedSegmenter:
            print('Unit testing segmenter network on fixed random segmentation')
            ## Re-initialize network weights in case any training has already happened
            segmenter.load_weights(logdir + '/segmenter_init.h5', by_name=True)

            segsProposal = trainSegmenterOnly(segmenter,
                                              Xs,
                                              Xs_mask,
                                              Y,
                                              trainIters,
                                              BATCH_SIZE,
                                              SEG_SHIFT)

            if ACOUSTIC:
                segsProposal[np.where(vad)] = 1.
            else:
                segsProposal[:, 0, ...] = 1.
            for doc in segsProposalXDoc:
                s, e = doc_indices[doc]
                segsProposalXDoc[doc] = segsProposal[s:e]
                if ACOUSTIC:
                    masked_proposal = np.ma.array(segsProposalXDoc[doc], mask=Xs_mask[s:e])
                    segsProposalXDoc[doc] = masked_proposal.compressed()

            randomTargetsXDoc = dict.fromkeys(doc_indices)

            print('Scoring network predictions')
            if ACOUSTIC:
                for doc in randomTargetsXDoc:
                    s, e = doc_indices[doc]
                    masked_target = np.ma.array(Y[s:e], mask=Xs_mask[s:e])
                    randomTargetsXDoc[doc] = masked_target.compressed()
                scores = getSegScores(frameSegs2timeSegs(intervals, randomTargetsXDoc),
                                      frameSegs2timeSegs(intervals, segsProposalXDoc), acoustic=ACOUSTIC)
            else:
                for doc in randomTargetsXDoc:
                    s, e = doc_indices[doc]
                    randomTargetsXDoc[doc] = charSeq2WrdSeq(Y[s:e], gold[doc])
                scores = getSegScores(randomTargetsXDoc, segsProposalXDoc, acoustic=ACOUSTIC)

            print('')
            print('Random segmentation score')
            printSegScores(scores, acoustic=ACOUSTIC)

        ## Gold word segmentations
        if not ACOUSTIC or GOLDWRD:
            if ACOUSTIC:
                _, goldseg = timeSegs2frameSegs(GOLDWRD)
                Y = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
            else:
                goldseg = gold
                Y = texts2Segs(gold, maxChar)
            if args.supervisedAE:
                print('Unit testing auto-encoder network on gold (word-level) segmentations')
                ## Re-initialize network weights in case any training has already happened
                ae_full.load_weights(logdir + '/model_init.h5', by_name=True)

                for i in range(trainIters):
                    print('Iteration %d' % (i + 1))
                    Xae = trainAEOnly(ae_full,
                                      Xs,
                                      Xs_mask,
                                      Y,
                                      maxUtt,
                                      maxLen,
                                      1,
                                      BATCH_SIZE,
                                      REVERSE_UTT,
                                      N_RESAMPLE)

                    plotPredsUtt(utt_ids,
                                 ae_full,
                                 Xae,
                                 getYae(Xae, REVERSE_UTT),
                                 logdir,
                                 'goldwrd',
                                 i+1,
                                 BATCH_SIZE,
                                 DEBUG)

                    if not ACOUSTIC:
                        printReconstruction(10, ae_full, Xae, ctable, BATCH_SIZE, REVERSE_UTT)


            if args.supervisedSegmenter:
                print('Unit testing segmenter network on gold (word-level) segmentations')
                ## Re-initialize network weights in case any training has already happened
                segmenter.load_weights(logdir + '/segmenter_init.h5', by_name=True)

                segsProposal = trainSegmenterOnly(segmenter,
                                                  Xs,
                                                  Xs_mask,
                                                  Y,
                                                  trainIters,
                                                  BATCH_SIZE,
                                                  SEG_SHIFT)

                if ACOUSTIC:
                    segsProposal[np.where(vad)] = 1.
                else:
                    segsProposal[:,0,...] = 1.
                for doc in segsProposalXDoc:
                    s, e = doc_indices[doc]
                    segsProposalXDoc[doc] = segsProposal[s:e]
                    if ACOUSTIC:
                        masked_proposal = np.ma.array(segsProposalXDoc[doc], mask=Xs_mask[s:e])
                        segsProposalXDoc[doc] = masked_proposal.compressed()

                print('Scoring segmentations')
                if ACOUSTIC:
                    scores = getSegScores(gold['wrd'], frameSegs2timeSegs(intervals, segsProposalXDoc), acoustic=ACOUSTIC)
                else:
                    scores = getSegScores(gold, segsProposalXDoc, acoustic=ACOUSTIC)
                printSegScores(scores, acoustic=ACOUSTIC)

        ## Gold phone segmentations
        if ACOUSTIC and GOLDPHN:
            _, goldseg = timeSegs2frameSegs(GOLDPHN)
            Y = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)

            if args.supervisedAE:
                print('Unit testing auto-encoder network on gold (phone-level) segmentations')
                ## Re-initialize network weights in case any training has already happened
                ae_full.load_weights(logdir + '/model_init.h5', by_name=True)

                for i in range(trainIters):
                    print('Iteration %d' % (i + 1))
                    Xae = trainAEOnly(ae_full,
                                      Xs,
                                      Xs_mask,
                                      Y,
                                      maxUtt,
                                      maxLen,
                                      1,
                                      BATCH_SIZE,
                                      REVERSE_UTT,
                                      N_RESAMPLE)

                    plotPredsUtt(utt_ids,
                                 ae_full,
                                 Xae,
                                 getYae(Xae, REVERSE_UTT),
                                 logdir,
                                 'goldphn',
                                 i+1,
                                 BATCH_SIZE,
                                 DEBUG)

            if args.supervisedSegmenter:
                print('Unit testing segmenter network on gold (phone-level) segmentations')
                ## Re-initialize network weights in case any training has already happened
                segmenter.load_weights(logdir + '/segmenter_init.h5', by_name=True)

                segsProposal = trainSegmenterOnly(segmenter,
                                                  Xs,
                                                  Xs_mask,
                                                  Y,
                                                  trainIters,
                                                  BATCH_SIZE,
                                                  SEG_SHIFT)

                segsProposal[np.where(vad)] = 1.
                for doc in segsProposalXDoc:
                    s, e = doc_indices[doc]
                    segsProposalXDoc[doc] = segsProposal[s:e]
                    masked_proposal = np.ma.array(segsProposalXDoc[doc], mask=Xs_mask[s:e])
                    segsProposalXDoc[doc] = masked_proposal.compressed()

                print('Scoring segmentations')
                scores = getSegScores(gold['phn'], frameSegs2timeSegs(intervals, segsProposalXDoc), acoustic=ACOUSTIC)
                printSegScores(scores, acoustic=ACOUSTIC)

        exit()




    ##################################################################################
    ##################################################################################
    ##
    ##  Pretrain auto-encoder on initial segmentation proposal
    ##
    ##################################################################################
    ##################################################################################

    print()
    if iteration < pretrainIters and pretrain:
        print("Starting auto-encoder pretraining")
    while iteration < pretrainIters and pretrain:
        print('-' * 50)
        print('Iteration', iteration + 1)

        segs = sampleSeg(pSegs)
        t0 = time.time()
        ## Randomly permute samples
        p, p_inv = getRandomPermutation(len(Xs))
        Xs = Xs[p]
        Xs_mask = Xs_mask[p]

        if AE_NET:
            ts0 = time.time()
            Xae, deletedChars, oneLetter = updateAE(ae_phon,
                                                    ae_utt,
                                                    embed_words,
                                                    Xs,
                                                    Xs_mask,
                                                    segs,
                                                    maxUtt,
                                                    maxLen,
                                                    BATCH_SIZE,
                                                    REVERSE_UTT,
                                                    N_RESAMPLE)
            ts1 = time.time()
            print('Split time: %.2fs' %(ts1-ts0))

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

            t1 = time.time()
            print('Auto-encoder input preprocessing completed in %.2fs.' %(t1-t0))

            Yae = getYae(Xae, REVERSE_UTT)

            if DEBUG and False:
                preds = ae_full.predict(Xae, batch_size=BATCH_SIZE)
                print('Finite prediction cells: %s' %np.isfinite(preds).sum())

            print("Total deleted chars:    %d" %(deletedChars.sum()))
            print("Total one-letter words: %d" %(oneLetter.sum()))

        if SEG_NET and not AE_NET:

            print('Training segmenter network on random segmentation.')
            updateSegmenter(segmenter,
                            Xs,
                            Xs_mask,
                            segs,
                            SEG_SHIFT,
                            BATCH_SIZE)

        if AE_NET:
            ae_full.fit(Xae,
                        Yae,
                        batch_size=BATCH_SIZE,
                        epochs=1)

            plotPredsUtt(utt_ids,
                         ae_full,
                         Xae,
                         Yae,
                         logdir,
                         'pretrain',
                         iteration,
                         BATCH_SIZE,
                         DEBUG)

            # Correctness checks for NN masking
            if DEBUG:
                out = ae_full.predict(Xae, batch_size=BATCH_SIZE)
                print('Timesteps in input: %s' %Xae.any(-1).sum())
                print('Timesteps in output: %s (should equal timesteps in input)' %out.any(-1).sum())
                print('Deleted timesteps: %s' %int(deletedChars.sum()))
                print('Timesteps + deleted: %s (should be %s)' % (out.any(-1).sum() + int(deletedChars.sum()), sum([raw_cts[doc] for doc in raw_cts])))
                print('')

            if not ACOUSTIC:
                preds = ae_full.predict(Xae[:10])
                printReconstruction(10, ae_full, Xae, ctable, BATCH_SIZE, REVERSE_UTT)

        ## Invert permutation
        Xs = Xs[p_inv]
        Xs_mask = Xs_mask[p_inv]

        iteration += 1
        if iteration == pretrainIters:
            pretrain = False
            iteration = 0

        if AE_NET:
            ae_full.save(logdir + '/model.h5')
        if SEG_NET and not AE_NET:
            segmenter.save(logdir + '/segmenter.h5')
        with open(logdir + '/checkpoint.obj', 'wb') as f:
            checkpoint['iteration'] = iteration
            checkpoint['pretrain'] = pretrain
            pickle.dump(checkpoint, f)

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

    # # Check character scoring
    # goldSegs = dict.fromkeys(gold.keys())
    # for doc in goldSegs:
    #     goldSegs[doc] = text2Segs(gold[doc], maxChar)
    # printSegScores(getSegScores(gold, goldSegs, ACOUSTIC), ACOUSTIC)





    ##################################################################################
    ##################################################################################
    ##
    ##  Train model to find optimal segmentation
    ##
    ##################################################################################
    ##################################################################################

    batch_num_global = checkpoint.get('batch_num_global', 0)
    if iteration < trainIters:
        print('Starting training')
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

        if AE_NET:
            epochAELoss = checkpoint.get('epochAELoss', 0)
            if not ACOUSTIC:
                epochAEAcc = checkpoint.get('epochAEAcc', 0)
            epochDel = checkpoint.get('epochDel', 0)
            epochOneL = checkpoint.get('epochOneL', 0)
        if SEG_NET:
            epochSegLoss = checkpoint.get('epochSegLoss', 0)
        segsProposal = checkpoint.get('segsProposal', [])
        segProbs = checkpoint.get('segProbs', [])
        epochSeg = checkpoint.get('epochSeg', 0)
        b =  checkpoint.get('b', 0)

        ## Randomly permute samples
        if b == 0:
            p, p_inv = getRandomPermutation(len(Xs))
        else:
            p, p_inv = checkpoint.get('permute', getRandomPermutation(len(Xs)))
        Xs = Xs[p]
        Xs_mask = Xs_mask[p]
        pSegs = pSegs[p]
        if ACOUSTIC:
            vad = vad[p]

        while b < len(Xs):
            bt0 = time.time()

            Xs_batch = Xs[b:b+SAMPLING_BATCH_SIZE]
            Xs_mask_batch = Xs_mask[b:b+SAMPLING_BATCH_SIZE]
            if ACOUSTIC:
                vad_batch = vad[b:b+SAMPLING_BATCH_SIZE]

            if iteration < trainNoSegIters:
                pSegs_batch = pSegs[b:b+SAMPLING_BATCH_SIZE]
            else:
                if SEG_NET:
                    seg_inputs = np.zeros((len(Xs_batch), maxChar+SEG_SHIFT, charDim))
                    seg_inputs[:,:maxChar,:] = Xs_batch
                    preds = predictSegmenter(segmenter,
                                             Xs_batch,
                                             Xs_mask_batch,
                                             SEG_SHIFT,
                                             BATCH_SIZE)
                    pSegs_batch = (1-INTERPOLATION_RATE) * preds + INTERPOLATION_RATE * .5 * np.ones_like(preds)
                    # pSegs_batch = pSegs_batch**2
                else:
                    pSegs_batch = pSegs[b:b + SAMPLING_BATCH_SIZE]
                    pSegs_batch = (1-INTERPOLATION_RATE) * pSegs_batch + INTERPOLATION_RATE * .5 * np.ones_like(pSegs_batch)

                ## Force segmentations where needed
                if ACOUSTIC:
                    pSegs_batch[np.where(vad_batch)] = 1.
                else:
                    pSegs_batch[:,0] = 1.

                ## Zero-out segmentation probability in padding regions
                pSegs_batch[np.where(Xs_mask_batch)] = 0.
                # print(preds)
                # print(INTERPOLATION_RATE * .5 * np.ones_like(preds))
                # print(pSegs_batch)

            st0 = time.time()
            scores_batch = np.zeros((len(Xs_batch), N_SAMPLES, maxUtt))
            penalties_batch = np.zeros_like(scores_batch)
            segSamples_batch = np.zeros((len(Xs_batch), N_SAMPLES, maxChar))
            print()
            print('Batch %d/%d' %((b+1)/SAMPLING_BATCH_SIZE+1, N_BATCHES))
            for s in range(N_SAMPLES):
                sys.stdout.write('\rSample %d/%d' %(s+1, N_SAMPLES))
                sys.stdout.flush()
                segs_batch = sampleSeg(pSegs_batch)
                segSamples_batch[:,s,:] = np.squeeze(segs_batch, -1)

                if AE_NET:
                    Xae_batch, deletedChars_batch, oneLetter_batch = XsSeg2Xae(Xs_batch,
                                                                               Xs_mask_batch,
                                                                               segs_batch,
                                                                               maxUtt,
                                                                               maxLen,
                                                                               N_RESAMPLE)

                    Yae_batch = getYae(Xae_batch, REVERSE_UTT)
                    input_batch = Xae_batch
                    target_batch = Yae_batch
                    scorerNetwork = ae_full

                    scores_batch[:, s, :] = scoreXUtt(scorerNetwork,
                                                      input_batch,
                                                      target_batch,
                                                      BATCH_SIZE,
                                                      REVERSE_UTT,
                                                      metric=METRIC)

                    penalties_batch[:,s,:] -= deletedChars_batch * DEL_WT
                    #print(deletedChars_batch)
                    #print(deletedChars_batch.shape)
                    #print(penalties_batch[:,s,:])
                    penalties_batch[:,s,:] -= oneLetter_batch * ONE_LETTER_WT
                    #print(oneLetter_batch)
                    #print(oneLetter_batch.shape)
                    #print(penalties_batch[:,s,:])
                    #raw_input()
                    if SEG_WT > 0:
                        for u in range(len(segs_batch)):
                            penalties_batch[u, s, -segs_batch[u].sum():] -= SEG_WT

            print()

            if AE_NET:
                print('Computing segmentation targets from samples')
                segProbs_batch, segsProposal_batch, n_Seg_batch = guessSegTargets(scores_batch,
                                                                                  penalties_batch,
                                                                                  segSamples_batch,
                                                                                  pSegs_batch,
                                                                                  Xs_mask_batch,
                                                                                  ALGORITHM,
                                                                                  maxLen,
                                                                                  DEL_WT,
                                                                                  ONE_LETTER_WT,
                                                                                  SEG_WT)
            else:
                segProbs_batch = np.expand_dims(segSamples_batch.sum(1) / segSamples_batch.shape[1], -1)
                segsProposal_batch = np.expand_dims(segProbs_batch > 0.5, -1)
                n_Seg_batch = segsProposal_batch.sum()

            st1 = time.time()
            print('Sampling time: %.2fs.' %(st1-st0))

            if AE_NET:
                Xae_batch, deletedChars_batch, oneLetter_batch = updateAE(ae_phon,
                                                                          ae_utt,
                                                                          embed_words,
                                                                          Xs_batch,
                                                                          Xs_mask_batch,
                                                                          segsProposal_batch,
                                                                          maxUtt,
                                                                          maxLen,
                                                                          BATCH_SIZE,
                                                                          REVERSE_UTT,
                                                                          N_RESAMPLE)

                Yae_batch = getYae(Xae_batch, REVERSE_UTT)

                print('Fitting full auto-encoder network')
                aeHist = ae_full.fit(Xae_batch,
                                     Yae_batch,
                                     batch_size=BATCH_SIZE,
                                     epochs=1)
            if SEG_NET:
                print('Fitting segmenter network')
                segHist = updateSegmenter(segmenter,
                                          Xs_batch,
                                          Xs_mask_batch,
                                          segProbs_batch,
                                          SEG_SHIFT,
                                          BATCH_SIZE)

            else:
                pSegs[b:b + SAMPLING_BATCH_SIZE] = segProbs_batch

            n_Char_batch = (1 - Xs_mask_batch).sum()
            print('Non-padding input timesteps in batch: %d' % n_Char_batch)
            print('Number of segments in batch: %d' % n_Seg_batch)
            if AE_NET:
                n_oneLetter_batch = oneLetter_batch.sum()
                n_deletedChars_batch = deletedChars_batch.sum()
                print('One-letter segments: %d' %n_oneLetter_batch)
                print('Deleted segments: %d' %n_deletedChars_batch)
            print('Mean segment length: %.2f' % (float(n_Char_batch) / n_Seg_batch))

            if AE_NET:
                epochAELoss += aeHist.history['loss'][-1]
                if not ACOUSTIC:
                    epochAEAcc += aeHist.history['masked_categorical_accuracy'][-1]
                epochDel += int(n_deletedChars_batch)
                epochOneL += int(n_oneLetter_batch)
            if SEG_NET:
                epochSegLoss += segHist.history['loss'][-1]

            # plotPredsSeg(range(N_VIZ),
            #              segmenter,
            #              Xs_batch,
            #              Xs_mask_batch,
            #              segProbs_batch,
            #              logdir,
            #              'debug',
            #              (b + 1) / SAMPLING_BATCH_SIZE + 1,
            #              SEG_SHIFT,
            #              BATCH_SIZE,
            #              DEBUG)

            segsProposal.append(segsProposal_batch)
            segProbs.append(segProbs_batch)
            epochSeg += int(n_Seg_batch)

            b += SAMPLING_BATCH_SIZE
            batch_num_global += 1

            if AE_NET:
                ae_full.save(logdir + '/model.h5')
            if SEG_NET:
                segmenter.save(logdir + '/segmenter.h5')
            with open(logdir + '/checkpoint.obj', 'wb') as f:
                checkpoint['permute'] = (p, p_inv)
                checkpoint['b'] = b
                checkpoint['batch_num_global'] = batch_num_global
                if AE_NET:
                    checkpoint['epochAELoss'] = epochAELoss
                    if not ACOUSTIC:
                        checkpoint['epochAEAcc'] = epochAEAcc
                    checkpoint['epochDel'] = epochDel
                    checkpoint['epochOneL'] = epochOneL
                if SEG_NET:
                    checkpoint['epochSegLoss'] = epochSegLoss
                checkpoint['iteration'] = iteration
                checkpoint['segsProposal'] = segsProposal
                checkpoint['segProbs'] = segProbs
                checkpoint['epochSeg'] = epochSeg
                pickle.dump(checkpoint, f)

            ## Evaluate on cross-validation set
            sig.disallow_interrupt()
            if crossValDir and (batch_num_global % EVAL_FREQ == 0):
                if ACOUSTIC:
                    otherParams = intervals_cv, GOLDWRD_CV, GOLDPHN_CV, vad_cv
                else:
                    otherParams = ctable_cv
                evalCrossVal(Xs_cv,
                             Xs_mask_cv,
                             gold_cv,
                             doc_list_cv,
                             doc_indices_cv,
                             utt_ids_cv,
                             otherParams,
                             maxLen,
                             maxUtt,
                             raw_total_cv,
                             logdir,
                             SEG_SHIFT,
                             BATCH_SIZE,
                             REVERSE_UTT,
                             iteration+1,
                             batch_num_global,
                             nResample = N_RESAMPLE,
                             ae_full = ae_full if AE_NET else None,
                             segmenter = segmenter if SEG_NET else None,
                             debug = DEBUG)
            sig.allow_interrupt()

            bt1 = time.time()
            print('Batch time: %.2fs' %(bt1-bt0))



        if AE_NET:
            epochAELoss /= N_BATCHES
            if not ACOUSTIC:
                epochAEAcc /= N_BATCHES
        if SEG_NET:
            epochSegLoss /= N_BATCHES
        segsProposal = np.concatenate(segsProposal)
        segProbs = np.concatenate(segProbs)

        ## Invert random permutation so evaluation aligns correctly
        Xs = Xs[p_inv]
        Xs_mask = Xs_mask[p_inv]
        if ACOUSTIC:
            vad = vad[p_inv]
        pSegs = pSegs[p_inv]
        segsProposal = segsProposal[p_inv]
        segProbs = segProbs[p_inv]

        iteration += 1
        b = 0

        # Evaluate on training data
        if AE_NET:
            n = 10
            Xae, _, _ = XsSeg2Xae(Xs,
                                  Xs_mask,
                                  segsProposal,
                                  maxUtt,
                                  maxLen,
                                  N_RESAMPLE)

            print('Plotting visualizations of auto-encoder output')
            plotPredsUtt(utt_ids,
                         ae_full,
                         Xae,
                         getYae(Xae, REVERSE_UTT, ACOUSTIC),
                         logdir,
                         'train',
                         iteration,
                         BATCH_SIZE,
                         DEBUG)

            if not ACOUSTIC:
                print('')
                print('Example reconstruction of learned segmentation')
                printReconstruction(10, ae_full, Xae, ctable, BATCH_SIZE, REVERSE_UTT)

        if SEG_NET:
            print('Plotting visualizations of segmenter output')
            plotPredsSeg(utt_ids,
                         segmenter,
                         Xs,
                         Xs_mask,
                         segProbs,
                         logdir,
                         'train',
                         iteration,
                         SEG_SHIFT,
                         BATCH_SIZE,
                         DEBUG)

        segsProposalXDoc = dict.fromkeys(doc_list)
        for doc in segsProposalXDoc:
            s,e = doc_indices[doc]
            segsProposalXDoc[doc] = segsProposal[s:e]
            if ACOUSTIC:
                masked_proposal = np.ma.array(segsProposalXDoc[doc], mask=Xs_mask[s:e])
                segsProposalXDoc[doc] = masked_proposal.compressed()

        sig.disallow_interrupt()
        segScore = writeLog(1,
                            iteration,
                            epochAELoss if AE_NET else None,
                            epochAEAcc if (not ACOUSTIC and AE_NET) else None,
                            epochSegLoss if SEG_NET else None,
                            epochDel if AE_NET else None,
                            epochOneL if AE_NET else None,
                            epochSeg,
                            gold,
                            segsProposalXDoc,
                            logdir,
                            intervals = intervals if ACOUSTIC else None,
                            acoustic = ACOUSTIC,
                            print_headers= not os.path.isfile(logdir + '/log_train.txt'),
                            filename= 'log_train.txt')
        sig.allow_interrupt()


        print('Total frames:', raw_total)
        if AE_NET:
            print('Auto-encoder loss:', epochAELoss)
            if not ACOUSTIC:
                print('Auto-encoder accuracy:', epochAEAcc)
            print('Deletions:', epochDel)
            print('One letter words:', epochOneL)
        if SEG_NET:
            print('Segmenter loss:', epochSegLoss)
        print('Total segmentation points:', epochSeg)

        sig.disallow_interrupt()
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
            writeTimeSegs(frameSegs2timeSegs(intervals, segsProposalXDoc), out_dir=logdir, TextGrid=False, dataset='train')
            writeTimeSegs(frameSegs2timeSegs(intervals, segsProposalXDoc), out_dir=logdir, TextGrid=True, dataset='train')
        else:
            printSegScores(getSegScores(gold, segsProposalXDoc, ACOUSTIC), ACOUSTIC)
            writeSolutions(logdir, segsProposalXDoc[doc_list[0]], gold[doc_list[0]], iteration, filename='seg_train.txt')


        if AE_NET:
            ae_full.save(logdir + '/model.h5')
        if SEG_NET:
            segmenter.save(logdir + '/segmenter.h5')
        with open(logdir + '/checkpoint.obj', 'wb') as f:
            checkpoint['iteration'] = iteration
            checkpoint['batch_num_global'] = batch_num_global
            checkpoint['b'] = b
            checkpoint['epochAELoss'] = 0
            checkpoint['epochSegLoss'] = 0
            checkpoint['epochDel'] = 0
            checkpoint['epochOneL'] = 0
            checkpoint['epochSeg'] = 0
            checkpoint['segsProposal'] = []
            checkpoint['segProbs'] = []
            pickle.dump(checkpoint, f)
        sig.allow_interrupt()

        it1 = time.time()
        print('Iteration time: %.2fs' %(it1-it0))

    print("Logs in", logdir)





    ##################################################################################
    ##################################################################################
    ##
    ## Compare AE loss for discovered vs. gold segmentation
    ##
    ##################################################################################
    ##################################################################################

    if AE_NET:

        print('Using discovered segmentation')

        printSegAnalysis(ae_full,
                         Xs,
                         Xs_mask,
                         segsProposal,
                         maxUtt,
                         maxLen,
                         METRIC,
                         BATCH_SIZE,
                         REVERSE_UTT,
                         ACOUSTIC)

        if ACOUSTIC:
            if GOLDWRD:
                _, goldseg = timeSegs2frameSegs(GOLDWRD)
                goldsegXUtt = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
            else:
                _, goldseg = timeSegs2frameSegs(GOLDPHN)
                goldsegXUtt = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
        else:
            goldsegXUtt = texts2Segs(gold, maxChar)


        Xae_gold, deletedChars_gold, oneLetter_gold = XsSeg2Xae(Xs,
                                                                Xs_mask,
                                                                goldsegXUtt,
                                                                maxUtt,
                                                                maxLen,
                                                                N_RESAMPLE)

        Yae_gold = getYae(Xae_gold, REVERSE_UTT)

        gold_lossXutt = scoreXUtt(ae_full,
                                  Xae_gold,
                                  Yae_gold,
                                  BATCH_SIZE,
                                  REVERSE_UTT,
                                  METRIC)

        gold_loss = gold_lossXutt.sum()
        gold_nChar = Xae_gold.any(-1).sum()

        print('Sum of losses for each word using gold segmentation%s: %.4f' %(' (word-level)' if (ACOUSTIC and GOLDWRD) else ' (phone-level) ' if (ACOUSTIC and GOLDPHN) else '', gold_loss))
        print('Deleted characters in segmentation: %d' % deletedChars_gold.sum())
        print('Input characterrs in segmentation: %d' % gold_nChar.sum())
        print('Loss per character: %.4f' %(float(gold_loss)/gold_nChar))
        print()

        if ACOUSTIC and GOLDWRD and GOLDPHN:
            _, goldseg = timeSegs2frameSegs(GOLDPHN)
            goldsegXUtt = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)

            Xae_gold, deletedChars_gold, oneLetter_gold = XsSeg2Xae(Xs,
                                                                    Xs_mask,
                                                                    goldsegXUtt,
                                                                    maxUtt,
                                                                    maxLen,
                                                                    N_RESAMPLE)

            Yae_gold = getYae(Xae_gold, REVERSE_UTT)

            gold_lossXutt = scoreXUtt(ae_full,
                                      Xae_gold,
                                      Yae_gold,
                                      BATCH_SIZE,
                                      REVERSE_UTT,
                                      METRIC)

            gold_loss = gold_lossXutt.sum()
            gold_nChar = Xae_gold.any(-1).sum()

            print('Sum of losses for each word using gold segmentation (phone-level): %.4f' % gold_loss)
            print('Deleted characters in segmentation: %d' % deletedChars_gold.sum())
            print('Input characterrs in segmentation: %d' % gold_nChar.sum())
            print('Loss per character: %.4f' %(float(gold_loss)/gold_nChar))
            print()

