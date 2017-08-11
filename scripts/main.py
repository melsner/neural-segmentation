from __future__ import print_function, division
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Model, Sequential
from keras import optimizers, metrics
from keras.layers import *
try:
    from keras.engine.training import slice_X
except:
    from keras.engine.training import _slice_arrays as slice_X
from keras import backend as K
import tensorflow as tf
from tensorflow.python.platform.test import is_gpu_available
import numpy as np
import numpy.ma as ma
import cPickle as pickle
import random
import sys
import re
import math
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
from unit_testing import *

class SigHandler(object):
    def __init__(self):
        self.sigint_received = False
        self.interrupt_allowed = True

    def disallow_interrupt(self):
        self.interrupt_allowed = False

    def allow_interrupt(self):
        self.interrupt_allowed = True
        if self.sigint_received == True:
            print()
            sys.exit(0)

    def interrupt(self, signum = None, frame = None):
        self.sigint_received = True
        if self.interrupt_allowed == True:
            print()
            sys.exit(0)

class Annealer(object):
    def __init__(self, t, r):
        self.t = t
        self.r = r

    def temp(self):
        return self.t

    def cool(self, steps=1):
        self.t = max(self.t - self.r*steps, 1)

    def step(self):
        return self.temp()
        self.cool()

sig = SigHandler()
signal.signal(signal.SIGINT, sig.interrupt)

argmax = lambda array: max(izip(array, xrange(len(array))))[1]
argmin = lambda array: min(izip(array, xrange(len(array))))[1]


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
    parser.add_argument("--aeType")
    parser.add_argument("--fitType")
    parser.add_argument("--vae")
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
    parser.add_argument("--coolingFactor")
    parser.add_argument("--annealTemp")
    parser.add_argument("--annealRate")
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
    parser.add_argument("--latentDim")
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

    loadModels = False
    if args.logfile == None:
        logdir = "logs/" + str(os.getpid()) + '/'
    else:
        logdir = "logs/" + args.logfile

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        if not (args.supervisedAE or args.supervisedAEPhon or args.supervisedSegmenter):
            loadModels = True

    if loadModels and os.path.exists(logdir + '/checkpoint.obj'):
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
    assert ALGORITHM in ['importance', 'importanceLocal', '1best', 'viterbi'], 'Algorithm "%s" is not supported. Use one of: %s' %(ALGORITHM, ', '.join['importance"', '1best', 'viterbi'])
    assert AE_NET or not ALGORITHM == 'viterbi', 'Viterbi sampling requires the AE network to be turned on.'
    OPTIM = args.optimizer if args.optimizer else checkpoint.get('optimizer', 'nadam' if ACOUSTIC else 'nadam')
    REVERSE_UTT = args.reverseUtt
    AE_TYPE = checkpoint.get('aeType', args.aeType if args.aeType else 'rnn' if ACOUSTIC else 'rnn')
    assert AE_TYPE in ['rnn', 'cnn'], 'The AE type requested (%s) is not supported (choose from "rnn", "cnn")' %AE_TYPE
    FIT_TYPE = args.fitType if args.fitType else checkpoint.get('fitType', 'parts' if ACOUSTIC else 'parts')
    assert FIT_TYPE in ['parts', 'whole', 'both'], 'If --fitType is specified, must be one of "parts", "whole", or "both" (defaults to "both")'
    FIT_FULL = FIT_TYPE in ['whole', 'both']
    FIT_PARTS = FIT_TYPE in ['parts', 'both']
    N_RESAMPLE = checkpoint.get('nResample', int(args.nResample) if args.nResample else None if ACOUSTIC else None)
    assert ACOUSTIC or not N_RESAMPLE, 'Resampling disallowed in character mode since it does not make any sense'
    crossValDir = args.crossValDir if args.crossValDir else checkpoint.get('crossValDir', None)
    EVAL_FREQ = int(args.evalFreq) if args.evalFreq else checkpoint.get('evalFreq', 10 if ACOUSTIC else 50)
    MASK_VALUE = 0 if ACOUSTIC else 1
    wordHidden = checkpoint.get('wordHidden', int(args.wordHidden) if args.wordHidden else 100 if ACOUSTIC else 80)
    uttHidden = checkpoint.get('uttHidden', int(args.uttHidden) if args.uttHidden else 500 if ACOUSTIC else 400)
    segHidden = checkpoint.get('segHidden', int(args.segHidden) if args.segHidden else 500 if ACOUSTIC else 100)
    wordDropout = checkpoint.get('wordDropout', float(args.wordDropout) if args.wordDropout else 0.25 if ACOUSTIC else 0.25)
    charDropout = checkpoint.get('charDropout', float(args.charDropout) if args.charDropout else 0.25 if ACOUSTIC else 0.5)
    COOLING_FACTOR = float(args.coolingFactor) if args.coolingFactor != None else checkpoint.get('coolingFactor', 1 if ACOUSTIC else 1)
    ANNEAL_TEMP = float(args.annealTemp) if args.annealTemp != None else checkpoint.get('annealTemp', 1 if ACOUSTIC else 1)
    ANNEAL_RATE = float(args.annealRate) if args.annealRate != None else checkpoint.get('annealRate', 0 if ACOUSTIC else 0)
    maxChar = checkpoint.get('maxChar', int(args.maxChar) if args.maxChar else 500 if ACOUSTIC else 30)
    maxUtt = checkpoint.get('maxUtt', int(args.maxUtt) if args.maxUtt else 50 if ACOUSTIC else 10)
    maxLen = checkpoint.get('maxLen', int(args.maxLen) if args.maxLen else 100 if ACOUSTIC else 7)
    DEPTH = checkpoint.get('depth', int(args.depth) if args.depth != None else 1)
    SEG_SHIFT = int(args.segShift) if args.segShift != None else checkpoint.get('segShift', 0 if ACOUSTIC else 0)
    pretrainIters = int(args.pretrainIters) if args.pretrainIters else checkpoint.get('pretrainIters', 10 if ACOUSTIC else 10)
    trainNoSegIters = int(args.trainNoSegIters) if args.trainNoSegIters else checkpoint.get('trainNoSegIters', 10 if ACOUSTIC else 10)
    trainIters = int(args.trainIters) if args.trainIters else checkpoint.get('trainIters', 100 if ACOUSTIC else 80)
    METRIC = args.metric if args.metric else checkpoint.get('metric', 'mse' if ACOUSTIC else 'logprob')
    DEL_WT = float(args.delWt) if args.delWt else checkpoint.get('delWt', 500 if ACOUSTIC else 50)
    ONE_LETTER_WT = float(args.oneLetterWt) if args.oneLetterWt else checkpoint.get('oneLetterWt', 500 if ACOUSTIC else 10)
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
    N_VIZ = checkpoint.get('nViz', int(args.nViz) if args.nViz != None else 10)
    VAE = bool(args.vae) if args.vae is not None else checkpoint.get('vae', 0 if ACOUSTIC else 0)
    LATENT_DIM = checkpoint.get('latentDim', int(args.latentDim) if args.latentDim else 5 if ACOUSTIC else 2)
    if SEG_NET and not AE_NET:
        METRIC = 'logprobbinary'

    annealer = Annealer(ANNEAL_TEMP, ANNEAL_RATE)




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
    checkpoint['aeType'] = AE_TYPE
    checkpoint['fitType'] = FIT_TYPE
    checkpoint['nResample'] = N_RESAMPLE
    checkpoint['crossValDir'] = crossValDir
    checkpoint['evalFreq'] = EVAL_FREQ
    checkpoint['wordHidden'] = wordHidden
    checkpoint['uttHidden'] = uttHidden
    checkpoint['segHidden'] = segHidden
    checkpoint['wordDropout'] = wordDropout
    checkpoint['charDropout'] = charDropout
    checkpoint['maxChar'] = maxChar
    checkpoint['coolingFactor'] = COOLING_FACTOR
    checkpoint['annealTemp'] = ANNEAL_TEMP
    checkpoint['annealRate'] = ANNEAL_RATE
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
    checkpoint['vae'] = VAE
    checkpoint['latentDim'] = LATENT_DIM





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
        print('  AE network type ("cnn", "rnn"): %s' % AE_TYPE, file=f)
        print('  Fitting type ("parts", "whole", or "both"): %s' % FIT_TYPE, file=f)
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
        print('  Segmenter output cooling factor: %s' % COOLING_FACTOR, file=f)
        print('  Loss annealing starting temperature: %s' % ANNEAL_TEMP, file=f)
        print('  Loss annealing rate: %s' % ANNEAL_RATE, file=f)
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
        print('  Variational auto-encoding: %s' % VAE, file=f)
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
              '--fitType' if args.fitType else '',
              '--nResample %s' % N_RESAMPLE if N_RESAMPLE else '',
              ('--crossValDir %s' % crossValDir) if crossValDir else '',
              '--aeType %s' % AE_TYPE,
              '--evalFreq %s' % EVAL_FREQ,
              '--algorithm %s' % ALGORITHM,
              '--optimizer %s' % OPTIM,
              '--wordHidden %s' % wordHidden,
              '--uttHidden %s' % uttHidden,
              '--segHidden %s' % segHidden,
              '--wordDropout %s' % wordDropout,
              '--charDropout %s' % charDropout,
              '--coolingFactor %s' % COOLING_FACTOR,
              '--annealTemp %s' % ANNEAL_TEMP,
              '--annealRate %s' % ANNEAL_RATE,
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
              '--vae %s' % int(VAE),
              '--latentDim %s' % int(LATENT_DIM),
              '--gpufrac %s' % args.gpufrac, file=f)

    print("Logging at", logdir)




    ##################################################################################
    ##################################################################################
    ##
    ##  Construct network graphs
    ##
    ##################################################################################
    ##################################################################################


    # ## Doesn't work, model won't save for some reason -- would be nice to not hvae this in the main script
    # ae, segmenter = constructNetworks(wordHidden,
    #                                   uttHidden,
    #                                   charDropout,
    #                                   wordDropout,
    #                                   maxChar,
    #                                   maxUtt,
    #                                   maxLen,
    #                                   charDim,
    #                                   segHidden,
    #                                   sess,
    #                                   logdir,
    #                                   aeNet=AE_NET,
    #                                   aeType=AE_TYPE,
    #                                   vae=VAE,
    #                                   latentDim=LATENT_DIM,
    #                                   segNet=SEG_NET,
    #                                   segShift=SEG_SHIFT,
    #                                   optim=OPTIM,
    #                                   reverseUtt=REVERSE_UTT,
    #                                   acoustic=ACOUSTIC,
    #                                   loadModels=loadModels)

    print("Constructing networks")

    RNN = recurrent.LSTM

    adam = optimizers.adam(clipnorm=1.)
    nadam = optimizers.Nadam(clipnorm=1.)
    rmsprop = optimizers.RMSprop(clipnorm=1.)
    optim_map = {'adam': adam, 'nadam': nadam, 'rmsprop': rmsprop}

    if AE_NET:
        ## AUTO-ENCODER NETWORK
        print('Using %s for auto-encoding' %(AE_TYPE.upper()))
        print()

        ## USEFUL VARIABLES
        wLen = N_RESAMPLE if N_RESAMPLE else maxLen
        wEmbDim = LATENT_DIM if VAE else wordHidden
        if AE_TYPE == 'cnn':
            x_wrd_pad = int(2 ** (math.ceil(math.log(wLen, 2))) - wLen)
            y_wrd_pad = int(2 ** (math.ceil(math.log(charDim, 2))) - charDim)
            x_utt_pad = int(2 ** (math.ceil(math.log(maxUtt, 2))) - maxUtt)
            y_utt_pad = int(2 ** (math.ceil(math.log(wEmbDim, 2))) - wEmbDim)
            expand = Lambda(lambda x: K.expand_dims(x, -1), name='ExpandDim')
            squeeze = Lambda(lambda x: K.squeeze(x, -1), name='SqueezeDim')
            nFilters = 10

        ## INPUTS
        fullInput = Input(shape=(maxUtt, wLen, charDim), name='FullInput')
        phonInput = Input(shape=(wLen, charDim), name='PhonInput')
        uttInput = Input(shape=(maxUtt, LATENT_DIM if VAE else wordHidden), name='UttInput')
        wordDecInput = Input(shape=(LATENT_DIM if VAE else wordHidden,), name='WordDecoderInput')
        uttDecInput = Input(shape=(uttHidden,), name='UttDecoderInput')

        ## OUTPUT MASKS
        m_phon2phon = (lambda x: x * K.cast(K.any(K.reverse(phonInput, 1), -1, keepdims=True), 'float32')) if REVERSE_UTT else \
                      (lambda x: x * K.cast(K.any(phonInput, -1, keepdims=True), 'float32'))
        m_utt2utt =   (lambda x: x * K.cast(K.any(K.reverse(uttInput, 1), -1, keepdims=True), 'float32')) if REVERSE_UTT else \
                      (lambda x: x * K.cast(K.any(uttInput, -1, keepdims=True), 'float32'))
        m_full2utt =  lambda x: x * K.cast(K.any(K.any(fullInput, -1), -1, keepdims=True), 'float32')
        m_full2uttR = (lambda x: x * K.cast(K.any(K.any(K.reverse(fullInput, (1, 2)), -1), -1, keepdims=True), 'float32')) if REVERSE_UTT else \
                      (lambda x: x * K.cast(K.any(K.any(fullInput, -1), -1, keepdims=True), 'float32'))
        m_full2full = (lambda x: x * K.cast(K.any(K.reverse(fullInput, (1, 2)), -1, keepdims=True), 'float32')) if REVERSE_UTT else \
                      (lambda x: x * K.cast(K.any(fullInput, -1, keepdims=True), 'float32'))

        ## WORD ENCODER
        wordEncoder = phonInput
        wordEncoder = Masking(mask_value=0.0, name='WordEncoderInputMask')(wordEncoder)
        # if not ACOUSTIC:
        #     wordEncoder = Embedding(charDim, charDim, name='CharacterEmbedding')(Lambda(lambda x: K.argmax(x))(wordEncoder))
        wordEncoder = Dropout(charDropout, noise_shape=(1, wLen, 1), name='CharacterDropout')(wordEncoder)
        if AE_TYPE == 'rnn':
            wordEncoder = RNN(wordHidden, name='WordEncoderRNN')(wordEncoder)
        elif AE_TYPE == 'cnn':
            wordEncoder = expand(wordEncoder)
            wordEncoder = ZeroPadding2D(padding=((0, x_wrd_pad), (0, y_wrd_pad)), name='WordInputPadder')(wordEncoder)
            wordEncoder = Conv2D(nFilters, (3, 3), padding='same', activation='elu', name='WordConv')(wordEncoder)
            wordEncoder = Conv2D(nFilters, (3, 3), strides=(2,2), padding='same', activation='elu', name='WordStrideConv')(wordEncoder)
            wordEncoder = Flatten(name='WordConvFlattener')(wordEncoder)
            wordEncoder = Dense(wordHidden, activation='elu', name='WordFullyConnected')(wordEncoder)
        if VAE:
            word_mean = Dense(LATENT_DIM, name='WordMean')(wordEncoder)
            word_log_var = Dense(LATENT_DIM, name='WordVariance')(wordEncoder)

            def sampling(args):
                word_mean, word_log_var = args
                epsilon = K.random_normal(shape=K.shape(word_mean), mean=0., stddev=1.0)
                return word_mean + K.exp(word_log_var/2) * epsilon
            wordEncoder = Lambda(sampling, output_shape=(LATENT_DIM,), name='WordEmbSampler')([word_mean, word_log_var])
        elif AE_TYPE == 'cnn':
            wordEncoder = Dense(wordHidden, name='WordEncoderOut')(wordEncoder)
        wordEncoder = Model(inputs=phonInput, outputs=wordEncoder, name='WordEncoder')

        ## WORDS ENCODER
        wordsEncoder = TimeDistributed(wordEncoder, name='WordEncoderDistributer')(fullInput)
        wordsEncoder = Lambda(m_full2utt, name='WordsEncoderPremask')(wordsEncoder)
        wordsEncoder = Model(inputs=fullInput, outputs=wordsEncoder, name='WordsEncoder')

        ## UTTERANCE ENCODER
        uttEncoder = Masking(mask_value=0.0, name='UttInputMask')(uttInput)
        uttEncoder = Dropout(wordDropout, noise_shape=(1, maxUtt, 1), name='WordDropout')(uttEncoder)
        if AE_TYPE == 'rnn':
            uttEncoder = RNN(uttHidden, return_sequences=False, name='UttEncoderRNN')(uttEncoder)
        elif AE_TYPE == 'cnn':
            uttEncoder = expand(uttEncoder)
            uttEncoder = ZeroPadding2D(padding=((0, x_utt_pad), (0, y_utt_pad)), name='UttInputPadder')(uttEncoder)
            uttEncoder = Conv2D(nFilters, (3, 3), padding='same', activation='elu', name='UttConv')(uttEncoder)
            uttEncoder = Conv2D(nFilters, (3, 3), strides=(2, 2), padding='same', activation='elu', name='UttStrideConv')(uttEncoder)
            uttEncoder = Flatten(name='UttConvFlattener')(uttEncoder)
            uttEncoder = Dense(uttHidden, activation='elu', name='UttFullyConnected')(uttEncoder)
            uttEncoder = Dense(uttHidden, name='UttEncoderOut')(uttEncoder)
        uttEncoder = Model(inputs=uttInput, outputs=uttEncoder, name='UttEncoder')

        ## UTTERANCE DECODER
        if AE_TYPE == 'rnn':
            uttDecoder = RepeatVector(maxUtt, input_shape=(uttHidden,), name='UttEmbeddingRepeater')(uttDecInput)
            uttDecoder = RNN(uttHidden, return_sequences=True, name='UttDecoderRNN')(uttDecoder)
            uttDecoder = TimeDistributed(Dense(wEmbDim), name='UttDecoderOut')(uttDecoder)
        elif AE_TYPE == 'cnn':
            uttDecoder = Dense(int((maxUtt + x_utt_pad) / 2 * (wEmbDim + y_utt_pad) / 2 * nFilters), activation='elu', name='UttDecoderDenseIn')(uttDecInput)
            uttDecoder = Reshape((int((maxUtt + x_utt_pad) / 2), int((wEmbDim + y_utt_pad) / 2), nFilters), name='UttDecoderReshape')(uttDecoder)
            uttDecoder = Conv2D(nFilters, (3, 3), padding='same', activation='elu', name='UttDeconv')(uttDecoder)
            uttDecoder = UpSampling2D((2, 2), name='UttUpsample')(uttDecoder)
            uttDecoder = Cropping2D(((0, x_utt_pad), (0, y_utt_pad)), name='UttOutCrop')(uttDecoder)
            uttDecoder = Conv2D(1, (3, 3), padding='same', activation='linear', name='UttDecoderOut')(uttDecoder)
            uttDecoder = squeeze(uttDecoder)
        uttDecoder = Model(inputs=uttDecInput, outputs=uttDecoder, name='UttDecoder')

        ## WORD DECODER
        if AE_TYPE == 'rnn':
            wordDecoder = RepeatVector(wLen, input_shape=(wEmbDim,), name='WordEmbeddingRepeater')(wordDecInput)
            wordDecoder = Masking(mask_value=0, name='WordDecoderInputMask')(wordDecoder)
            wordDecoder = RNN(wordHidden, return_sequences=True, name='WordDecoderRNN')(wordDecoder)
            wordDecoder = TimeDistributed(Dense(charDim), name='WordDecoderDistributer')(wordDecoder)
            wordDecoder = Activation('linear' if ACOUSTIC else 'softmax', name='WordDecoderOut')(wordDecoder)
        elif AE_TYPE == 'cnn':
            wordDecoder = Dense(int((wLen + x_wrd_pad) / 2 * (charDim + y_wrd_pad) / 2 * nFilters), activation='elu', name='WordDecoderDenseIn')(wordDecInput)
            wordDecoder = Reshape((int((wLen + x_wrd_pad) / 2), int((charDim + y_wrd_pad) / 2), nFilters), name='WordDecoderReshape')(wordDecoder)
            wordDecoder = Conv2D(nFilters, (3, 3), padding='same', activation='elu', name='WordDeconv')(wordDecoder)
            wordDecoder = UpSampling2D((2,2), name='WordUpsample')(wordDecoder)
            wordDecoder = Cropping2D(((0, x_wrd_pad), (0, y_wrd_pad)), name='WordOutCrop')(wordDecoder)
            wordDecoder = Conv2D(1, (3, 3), padding='same')(wordDecoder)
            wordDecoder = squeeze(wordDecoder)
            if not ACOUSTIC:
                wordDecoder = TimeDistributed(Dense(charDim, activation='softmax'))(wordDecoder)
        wordDecoder = Model(inputs=wordDecInput, outputs=wordDecoder, name='WordDecoder')

        ## WORDS DECODER (OUTPUT LAYER)
        wordsDecoder = TimeDistributed(wordDecoder, name='WordsDecoderDistributer')(uttInput)
        wordsDecoder = Masking(mask_value=0.0, name='WordsDecoderInputMask')(wordsDecoder)
        wordsDecoder = Model(inputs=uttInput, outputs=wordsDecoder, name='WordsDecoder')

        ## ENCODER-DECODER LAYERS
        wordEncoderTensor = wordEncoder(phonInput)
        wordsEncoderTensor = wordsEncoder(fullInput)
        wordDecoderTensor = Masking(mask_value=0.0)(Lambda(m_phon2phon)(wordDecoder(wordDecInput)))
        wordsDecoderTensor = Masking(mask_value=0.0)(Lambda(m_full2full)(wordsDecoder(uttInput)))

        phonEncoderDecoder = wordDecoder(wordEncoderTensor)
        phonEncoderDecoder = Lambda(m_phon2phon, name='PhonPremask')(phonEncoderDecoder)

        uttEncoderDecoder = uttDecoder(uttEncoder(uttInput))
        uttEncoderDecoder = Model(inputs=uttInput, outputs=uttEncoderDecoder, name='UttEncoderDecoder')

        fullEncoderUttDecoder = uttEncoderDecoder(wordsEncoderTensor)
        fullEncoderUttDecoder = Lambda(m_full2uttR, name='UttPremask')(fullEncoderUttDecoder)

        fullEncoderDecoder = wordsDecoder(fullEncoderUttDecoder)
        fullEncoderDecoder = Lambda(m_full2full, name='FullPremask')(fullEncoderDecoder)

        ## VAE LOSS
        if VAE:
            def vae_loss(y_true, y_pred):
                loss_func = metrics.mean_squared_error if ACOUSTIC else masked_categorical_crossentropy
                ae_loss = loss_func(y_true, y_pred)
                ## We keep dims to tile the kl_loss out to all reconstructed characters/frames
                kl_loss = - 0.5 * K.mean(1 + word_log_var - K.square(word_mean) - K.exp(word_log_var), axis=-1,
                                         keepdims=True)
                return ae_loss + kl_loss

        ## COMPILED (TRAINABLE) MODELS
        ae_phon = Masking(mask_value=0, name='PhonMask')(phonEncoderDecoder)
        ae_phon = Model(inputs=phonInput, outputs=ae_phon, name='AEPhon')
        ae_phon.compile(
            loss=vae_loss if VAE else "mean_squared_error" if ACOUSTIC else masked_categorical_crossentropy,
            metrics=None if ACOUSTIC else [masked_categorical_accuracy],
            optimizer=optim_map[OPTIM])

        ae_utt = Masking(mask_value=0, name='UttMask')(Lambda(m_utt2utt, name='UttPremask')(uttEncoderDecoder(uttInput)))
        ae_utt = Model(inputs=uttInput, outputs=ae_utt, name='AEUtt')
        ae_utt.compile(loss="mean_squared_error", optimizer=optim_map[OPTIM])

        ae_full = Masking(mask_value=0, name='AEFullMask')(fullEncoderDecoder)
        ae_full = Model(inputs=fullInput, outputs=ae_full, name='AEFull')
        ae_full.compile(loss="mean_squared_error" if ACOUSTIC else masked_categorical_crossentropy,
                        metrics=None if ACOUSTIC else [masked_categorical_accuracy],
                        optimizer=optim_map[OPTIM])

        ## EMBEDDING/DECODING FEEDFORWARD SUB-NETWORKS
        ## Embeddings must be custom Keras functions instead of models to allow predict with and without dropout
        embed_word = K.function(inputs=[phonInput, K.learning_phase()], outputs=[wordEncoderTensor], name='EmbedWord')
        embed_word = makeFunction(embed_word)

        embed_words = K.function(inputs=[fullInput, K.learning_phase()], outputs=[wordsEncoderTensor], name='EmbedWords')
        embed_words = makeFunction(embed_words)

        embed_words_reconst = K.function(inputs=[fullInput, K.learning_phase()], outputs=[fullEncoderUttDecoder], name='EmbedWordsReconstructed')
        embed_words_reconst = makeFunction(embed_words_reconst)

        word_decoder = Model(inputs=[wordDecInput,phonInput], outputs=wordDecoderTensor, name='WordDecoder')
        word_decoder.compile(loss="mean_squared_error" if ACOUSTIC else masked_categorical_crossentropy,
                             metrics=None if ACOUSTIC else [masked_categorical_accuracy],
                             optimizer=optim_map[OPTIM])

        words_decoder = Model(inputs=[uttInput,fullInput], outputs=wordsDecoderTensor, name='WordsDecoder')
        words_decoder.compile(loss="mean_squared_error" if ACOUSTIC else masked_categorical_crossentropy,
                              metrics=None if ACOUSTIC else [masked_categorical_accuracy],
                              optimizer=optim_map[OPTIM])

        ## (SUB-)NETWORK SUMMARIES
        print('\n')
        print('='*50)
        print('(Sub-)Model Summaries:')
        print('='*50)
        print('\n')
        print('Word encoder model:')
        wordEncoder.summary()
        print('\n')
        print('Word decoder model:')
        wordDecoder.summary()
        print('\n')
        print('Utterance encoder model:')
        uttEncoder.summary()
        print('\n')
        print('Utterance decoder model:')
        uttDecoder.summary()
        print('\n')
        print('Phonological auto-encoder model:')
        ae_phon.summary()
        print('\n')
        print('Utterance auto-encoder model:')
        ae_utt.summary()
        print('\n')
        print('Full auto-encoder model:')
        ae_full.summary()
        print('\n')

        ## Initialize AE wrapper object containing all sub nets for convenience
        ae = AE(ae_full, ae_utt, ae_phon, embed_word, embed_words, embed_words_reconst, word_decoder, words_decoder)

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
        segmenter = Model(inputs=[segInput, segMaskInput], outputs= segmenter)
        segmenter.compile(loss="binary_crossentropy",
                          optimizer=optim_map[OPTIM])
        print('Segmenter network summary')
        segmenter.summary()
        print('')

        ## Initialize Segmenter wrapper object for convenience
        segmenter = Segmenter(segmenter, SEG_SHIFT)

    ## SAVE/LOAD WEIGHTS
    if AE_NET:
        ## Save current model weights for reinitalization if needed
        ae.save(logdir + '/ae_init.h5')
        if loadModels and os.path.exists(logdir + '/ae.h5'):
            print('Autoencoder checkpoint found. Loading weights...')
            ae.load(logdir + '/ae.h5')
        else:
            print('No autoencoder checkpoint found. Keeping default initialization.')
            ae.save(logdir + '/ae.h5')
    if SEG_NET:
        ## Save current model weights for reinitalization if needed
        segmenter.save(logdir + '/segmenter_init.h5')

        if loadModels and os.path.exists(logdir + '/segmenter.h5'):
            print('Segmenter checkpoint found. Loading weights...')
            segmenter.load(logdir + '/segmenter.h5')
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

        # Random segmentations
        testUnits(Xs,
                  Xs_mask,
                  pSegs,
                  maxChar,
                  maxUtt,
                  maxLen,
                  doc_indices,
                  utt_ids,
                  logdir,
                  reverseUtt=REVERSE_UTT,
                  batch_size=BATCH_SIZE,
                  nResample=N_RESAMPLE,
                  trainIters=trainIters,
                  supervisedAE=args.supervisedAE,
                  supervisedAEPhon=args.supervisedAEPhon,
                  supervisedSegmenter=args.supervisedSegmenter,
                  ae=ae if AE_NET else None,
                  segmenter=segmenter if SEG_NET else None,
                  vadBreaks=vadBreaks if ACOUSTIC else None,
                  vad=vad if ACOUSTIC else None,
                  intervals=intervals if ACOUSTIC else None,
                  ctable=None if ACOUSTIC else ctable,
                  goldEval=None if ACOUSTIC else gold,
                  segLevel='rand',
                  acoustic=ACOUSTIC,
                  fitFull=FIT_FULL,
                  fitParts=FIT_PARTS,
                  debug=DEBUG)

        ## Gold segmentations (phone if acoustic)
        if not ACOUSTIC or GOLDPHN:
            if ACOUSTIC:
                print('Using phone-level segmentations')
            testUnits(Xs,
                      Xs_mask,
                      pSegs,
                      maxChar,
                      maxUtt,
                      maxLen,
                      doc_indices,
                      utt_ids,
                      logdir,
                      reverseUtt=REVERSE_UTT,
                      batch_size=BATCH_SIZE,
                      nResample=N_RESAMPLE,
                      trainIters=trainIters,
                      supervisedAE=args.supervisedAE,
                      supervisedAEPhon=args.supervisedAEPhon,
                      supervisedSegmenter=args.supervisedSegmenter,
                      ae=ae if AE_NET else None,
                      segmenter=segmenter if SEG_NET else None,
                      vadBreaks=vadBreaks if ACOUSTIC else None,
                      vad=vad if ACOUSTIC else None,
                      intervals=intervals if ACOUSTIC else None,
                      ctable=None if ACOUSTIC else ctable,
                      gold=GOLDPHN if ACOUSTIC else gold,
                      goldEval=gold['phn'] if ACOUSTIC else gold,
                      segLevel='phn' if ACOUSTIC else 'gold',
                      acoustic=ACOUSTIC,
                      fitFull=FIT_FULL,
                      fitParts=FIT_PARTS,
                      debug=DEBUG)

        ## Gold word segmentations (if acoustic)
        if ACOUSTIC and GOLDWRD:
            if ACOUSTIC:
                print('Using word-level segmentations')
            testUnits(Xs,
                      Xs_mask,
                      pSegs,
                      maxChar,
                      maxUtt,
                      maxLen,
                      doc_indices,
                      utt_ids,
                      logdir,
                      reverseUtt=REVERSE_UTT,
                      batch_size=BATCH_SIZE,
                      nResample=N_RESAMPLE,
                      trainIters=trainIters,
                      supervisedAE=args.supervisedAE,
                      supervisedAEPhon=args.supervisedAEPhon,
                      supervisedSegmenter=args.supervisedSegmenter,
                      ae=ae if AE_NET else None,
                      segmenter=segmenter if SEG_NET else None,
                      vadBreaks=vadBreaks if ACOUSTIC else None,
                      vad=vad if ACOUSTIC else None,
                      intervals=intervals if ACOUSTIC else None,
                      ctable=None if ACOUSTIC else ctable,
                      gold=GOLDWRD if ACOUSTIC else gold,
                      goldEval=gold['wrd'] if ACOUSTIC else gold,
                      segLevel='wrd',
                      acoustic=ACOUSTIC,
                      fitFull=FIT_FULL,
                      fitParts=FIT_PARTS,
                      debug=DEBUG)

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

        t0 = time.time()

        # ## Use for testing with pre-training on gold segs
        # if ACOUSTIC:
        #     _, goldseg = timeSegs2frameSegs(GOLDPHN)
        #     segs = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
        # else:
        #     segs = texts2Segs(gold, maxChar)
        segs = sampleSeg(pSegs)

        if AE_NET:
            Xae, deletedChars, oneLetter, AEhist = ae.update(Xs,
                                                             Xs_mask,
                                                             segs,
                                                             maxUtt,
                                                             maxLen,
                                                             reverseUtt=REVERSE_UTT,
                                                             batch_size=BATCH_SIZE,
                                                             nResample=N_RESAMPLE,
                                                             nEpoch=1,
                                                             fitParts=FIT_PARTS,
                                                             fitFull=FIT_FULL)

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

            Yae = getYae(Xae, REVERSE_UTT)

            if DEBUG and False:
                preds = ae.predict(Xae, batch_size=BATCH_SIZE)
                print('Finite prediction cells: %s' %np.isfinite(preds).sum())

            print("Total deleted chars:    %d" %(deletedChars.sum()))
            print("Total one-letter words: %d" %(oneLetter.sum()))

        if SEG_NET and not AE_NET:

            print('Training segmenter network on random segmentation.')
            segmenter.update(Xs,
                             Xs_mask,
                             segs,
                             batch_size=BATCH_SIZE)

        if AE_NET:
            if N_RESAMPLE:
                Xae_full, _, _ = XsSeg2Xae(Xs,
                                           Xs_mask,
                                           segs,
                                           maxUtt,
                                           maxLen,
                                           nResample=None)

            ae.plotFull(utt_ids,
                        Xae_full if N_RESAMPLE else Xae,
                        getYae(Xae, REVERSE_UTT),
                        logdir,
                        'pretrain',
                        iteration+1,
                        batch_size=BATCH_SIZE,
                        Xae_resamp = Xae if N_RESAMPLE else None,
                        debug=DEBUG)

            # Correctness checks for NN masking
            if DEBUG:
                out = ae.predict(Xae, batch_size=BATCH_SIZE)
                print('Timesteps in input: %s' % Xae.any(-1).sum())
                print('Timesteps in output: %s (should equal timesteps in input)' % out.any(-1).sum())
                print('Deleted timesteps: %s' % int(deletedChars.sum()))
                print('Timesteps + deleted: %s (should be %s)' % (out.any(-1).sum() + int(deletedChars.sum()), sum([raw_cts[doc] for doc in raw_cts])))
                print('')

        if AE_NET:
            if not ACOUSTIC:
                printReconstruction(utt_ids, ae, Xae, ctable, batch_size=BATCH_SIZE, reverseUtt=REVERSE_UTT)

        iteration += 1
        if iteration == pretrainIters:
            pretrain = False
            iteration = 0

        if AE_NET:
            ae.save(logdir + '/ae.h5')
        if SEG_NET and not AE_NET:
            segmenter.save(logdir + '/segmenter.h5')
        with open(logdir + '/checkpoint.obj', 'wb') as f:
            checkpoint['iteration'] = iteration
            checkpoint['pretrain'] = pretrain
            pickle.dump(checkpoint, f)

        t1 = time.time()
        print('Iteration time: %.2fs' %(t1-t0))
        print()

    ## Pretraining finished, update checkpoint
    print('Saving restore point')
    with open(logdir + '/checkpoint.obj', 'wb') as f:
        checkpoint['pretrain'] = pretrain
        checkpoint['iteration'] = iteration
        pickle.dump(checkpoint, f)

    segScores = {'wrd': dict.fromkeys(doc_list), 'phn': dict.fromkeys(doc_list)}
    segScores['wrd']['##overall##'] = [(0,0,0), None, (0,0,0), None]
    segScores['phn']['##overall##'] = [(0,0,0), None, (0,0,0), None]
    print()



    ##################################################################################
    ##################################################################################
    ##
    ##  Train model to find optimal segmentation
    ##
    ##################################################################################
    ##################################################################################

    N_BATCHES = math.ceil(len(Xs)/SAMPLING_BATCH_SIZE)
    batch_num_global = checkpoint.get('batch_num_global', 0)

    if iteration < trainIters:
        print('Starting training')

    while iteration < trainIters:

        # if iteration % 10 == 0:
        #     print('Re-starting segmenter network')
        #     segmenter.load_weights(logdir + '/segmenter_init.h5', by_name=True)

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
                    preds = segmenter.predict(Xs_batch,
                                              Xs_mask_batch,
                                              batch_size=BATCH_SIZE)
                    pSegs_batch = (1-INTERPOLATION_RATE) * preds + INTERPOLATION_RATE * .5 * np.ones_like(preds)
                    pSegs_batch = pSegs_batch**COOLING_FACTOR
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

            st0 = time.time()
            scores_batch = np.zeros((len(Xs_batch), N_SAMPLES, maxUtt))
            penalties_batch = np.zeros_like(scores_batch)
            segSamples_batch = np.zeros((len(Xs_batch), N_SAMPLES, maxChar))
            print()
            print('Batch %d/%d' %((b+1)/SAMPLING_BATCH_SIZE+1, N_BATCHES))
            for s in range(N_SAMPLES):
                sys.stdout.write('\rSample %d/%d' %(s+1, N_SAMPLES))
                sys.stdout.flush()
                if iteration < trainNoSegIters:
                    segs_batch = sampleSeg(pSegs_batch)
                else:
                    segs_batch = sampleSeg(pSegs_batch, acoustic=ACOUSTIC, resamplePSegs=ACOUSTIC)
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
                    scorerNetwork = ae

                    scores_batch[:, s, :] = scoreXUtt(scorerNetwork,
                                                      input_batch,
                                                      target_batch,
                                                      reverseUtt=REVERSE_UTT,
                                                      batch_size=BATCH_SIZE,
                                                      agg='sum' if ALGORITHM=='viterbi' else 'mean',
                                                      metric=METRIC)

                    penalties_batch[:,s,:] -= deletedChars_batch * DEL_WT
                    penalties_batch[:,s,:] -= oneLetter_batch * ONE_LETTER_WT
                    if SEG_WT > 0:
                        for u in range(len(segs_batch)):
                            penalties_batch[u, s, -segs_batch[u].sum():] -= SEG_WT

            print()

            if AE_NET:
                print('Computing segmentation targets from samples')
                segProbs_batch, segsProposal_batch = guessSegTargets(scores_batch,
                                                                     penalties_batch,
                                                                     segSamples_batch,
                                                                     pSegs_batch,
                                                                     Xs_mask_batch,
                                                                     algorithm=ALGORITHM,
                                                                     maxLen=maxLen,
                                                                     delWt=DEL_WT,
                                                                     oneLetterWt=ONE_LETTER_WT,
                                                                     segWt=SEG_WT,
                                                                     annealer=annealer,
                                                                     acoustic=ACOUSTIC)
            else:
                segProbs_batch = np.expand_dims(segSamples_batch.sum(1) / segSamples_batch.shape[1], -1)
                segsProposal_batch = np.expand_dims(segProbs_batch > 0.5, -1)

            st1 = time.time()
            print('Sampling time: %.2fs.' %(st1-st0))

            segProbs_batch = segProbs_batch ** COOLING_FACTOR
            ## Get segmentations from segmenter network
            if SEG_NET:
                print('Fitting segmenter network to sampled targets')
                segHist = segmenter.update(Xs_batch,
                                           Xs_mask_batch,
                                           segProbs_batch,
                                           batch_size=BATCH_SIZE)
                # print('Getting segmentation proposal from network')
                # preds = segmenter.predict(Xs_batch,
                #                           Xs_mask_batch,
                #                           batch_size=BATCH_SIZE)
                # segsProposal_batch = pSegs2Segs(preds, ACOUSTIC)
                if ACOUSTIC:
                    segsProposal_batch[np.where(vad_batch)] = 1.
                else:
                    segsProposal_batch[:, 0, ...] = 1.
                segsProposal_batch[np.where(Xs_mask_batch)] = 0.
            else:
                pSegs[b:b + SAMPLING_BATCH_SIZE] = segProbs_batch
            n_Seg_batch = segsProposal_batch.sum()

            if AE_NET:
                print('Fitting AE network(s) using segmentation proposal')
                Xae_batch, deletedChars_batch, oneLetter_batch, AEhist = ae.update(Xs_batch,
                                                                                   Xs_mask_batch,
                                                                                   segsProposal_batch,
                                                                                   maxUtt,
                                                                                   maxLen,
                                                                                   reverseUtt=REVERSE_UTT,
                                                                                   batch_size=BATCH_SIZE,
                                                                                   nResample=N_RESAMPLE,
                                                                                   nEpoch=1,
                                                                                   fitParts=FIT_PARTS,
                                                                                   fitFull=FIT_FULL)

                Yae_batch = getYae(Xae_batch, REVERSE_UTT)

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
                epochAELoss += AEhist[0]
                if not ACOUSTIC:
                    epochAEAcc += AEhist[1]
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

            print('Saving restore point')
            if AE_NET:
                ae.save(logdir + '/ae.h5')
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
                             iteration+1,
                             batch_num_global,
                             reverseUtt=REVERSE_UTT,
                             batch_size=BATCH_SIZE,
                             nResample = N_RESAMPLE,
                             acoustic = ACOUSTIC,
                             ae = ae if AE_NET else None,
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

        # Evaluate on training data
        print('Scoring segmentations on training set')
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

        print('Plotting visualizations on training set')
        if AE_NET:
            n = 10
            Xae, _, _ = XsSeg2Xae(Xs,
                                  Xs_mask,
                                  segsProposal,
                                  maxUtt,
                                  maxLen,
                                  N_RESAMPLE)

            if N_RESAMPLE:
                Xae_full, _, _ = XsSeg2Xae(Xs,
                                           Xs_mask,
                                           segsProposal,
                                           maxUtt,
                                           maxLen,
                                           nResample=None)

            ae.plotFull(utt_ids,
                        Xae_full if N_RESAMPLE else Xae,
                        getYae(Xae, REVERSE_UTT),
                        logdir,
                        'train',
                        iteration,
                        batch_size=BATCH_SIZE,
                        Xae_resamp = Xae if N_RESAMPLE else None,
                        debug=DEBUG)

            if not ACOUSTIC:
                print('')
                print('Example reconstruction of learned segmentation')
                printReconstruction(utt_ids, ae, Xae, ctable, BATCH_SIZE, REVERSE_UTT)

        if SEG_NET:
            segmenter.plot(utt_ids,
                           Xs,
                           Xs_mask,
                           segProbs,
                           logdir,
                           'train',
                           iteration,
                           batch_size=BATCH_SIZE)

        print('Saving restore point')
        if AE_NET:
            ae.save(logdir + '/ae.h5')
        if SEG_NET:
            segmenter.save(logdir + '/segmenter.h5')
        with open(logdir + '/checkpoint.obj', 'wb') as f:
            checkpoint['iteration'] = iteration
            checkpoint['batch_num_global'] = batch_num_global
            if iteration < trainIters:
                ## Reset accumulators
                checkpoint['b'] = 0
                checkpoint['epochAELoss'] = 0
                if not ACOUSTIC:
                    checkpoint['epochAEAcc'] = 0
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


    print('Comparing network losses on discovered vs. gold segmentations')
    if AE_NET:

        print('Discovered segmentation')

        printSegAnalysis(ae,
                         Xs,
                         Xs_mask,
                         segsProposal,
                         maxUtt,
                         maxLen,
                         reverseUtt=REVERSE_UTT,
                         batch_size=BATCH_SIZE,
                         acoustic=ACOUSTIC)

        if ACOUSTIC:
            if GOLDWRD:
                print('Gold word segmentations')
                _, goldseg = timeSegs2frameSegs(GOLDWRD)
                goldsegXUtt = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
            else:
                print('Gold phone segmentations')
                _, goldseg = timeSegs2frameSegs(GOLDPHN)
                goldsegXUtt = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
        else:
            print('Gold word segmentations')
            goldsegXUtt = texts2Segs(gold, maxChar)

        printSegAnalysis(ae,
                         Xs,
                         Xs_mask,
                         goldsegXUtt,
                         maxUtt,
                         maxLen,
                         reverseUtt=REVERSE_UTT,
                         batch_size=BATCH_SIZE,
                         acoustic=ACOUSTIC)


        if ACOUSTIC and GOLDWRD and GOLDPHN:
            print('Gold phone segmentations')
            _, goldseg = timeSegs2frameSegs(GOLDPHN)
            goldsegXUtt = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)

            printSegAnalysis(ae,
                             Xs,
                             Xs_mask,
                             goldsegXUtt,
                             maxUtt,
                             maxLen,
                             reverseUtt=REVERSE_UTT,
                             batch_size=BATCH_SIZE,
                             acoustic=ACOUSTIC)

