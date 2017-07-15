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
from itertools import izip
import matplotlib.pyplot as plt
from collections import defaultdict
from echo_words import CharacterTable, pad
from capacityStatistics import getPseudowords
from ae_io import *
from data_handling import *
from sampling import *
from scoring import *

argmax = lambda array: max(izip(array, xrange(len(array))))[1]
argmin = lambda array: min(izip(array, xrange(len(array))))[1]

adam = optimizers.adam(clipnorm=1.)
nadam = optimizers.Nadam(clipnorm=1.)
rmsprop = optimizers.RMSprop(clipnorm=1.)
optim_map = {'adam': adam, 'nadam': nadam, 'rmsprop': rmsprop}

def inputSliceClosure(dim):
    return lambda xx: xx[:, dim, :, :]

def outputSliceClosure(dim):
    return lambda xx: xx[:, dim, :]

def mask_output(x, input, mask_value, reverseUtt):
    if reverseUtt:
        m = K.any(K.not_equal(K.reverse(input, (1,2)), mask_value), axis=-1, keepdims=True)
    else:
        m = K.any(K.not_equal(input, mask_value), axis=-1, keepdims=True)
    x *= K.cast(m, 'float32')
    x += 1 - K.cast(m, 'float32')
    return x


def masked_categorical_crossentropy(y_true, y_pred):
    mask = K.cast(K.expand_dims(K.any(y_true, -1), axis=-1), 'float32')
    y_pred *= mask
    y_pred += 1-mask
    y_pred += 1-mask
    losses = K.categorical_crossentropy(y_pred, y_true)
    losses *= K.squeeze(mask, -1)
    ## Normalize by number of real segments, using a small non-zero denominator in cases of padding characters
    ## in order to avoid division by zero
    #losses /= (K.mean(mask) + (1e-10*(1-K.mean(mask))))
    return losses

def masked_mean_squared_error(y_true, y_pred):
    y_pred = y_pred * K.cast(K.any(K.reverse(y_true, (1,2)), axis=-1, keepdims=True), 'float32')
    return K.mean(K.square(y_pred - y_true), axis=-1)

def masked_categorical_accuracy(y_true, y_pred):
    mask = K.cast(K.expand_dims(K.greater(K.argmax(y_true, axis=-1), 0), axis=-1), 'float32')
    accuracy = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'float32')
    accuracy *= K.squeeze(mask, -1)
    ## Normalize by number of real segments, using a small non-zero denominator in cases of padding characters
    ## in order to avoid division by zero
    #accuracy /= (K.mean(mask) + (1e-10*(1-K.mean(mask))))
    return accuracy

def wrdEmbdTargets(wrd_emb, Xae, batch_size, reverseUtt):
    if reverseUtt:
        get_embds = lambda: np.flip(wrd_emb.predict(Xae, batch_size=batch_size), 1)
    else:
        get_embds = lambda: wrd_emb.predict(Xae, batch_size=batch_size)
    return get_embds()

def updateAE(ae_phon, ae_utt, Xs, Xs_mask, segs, maxUtt, maxLen, batch_size, reverseUtt):
    ## Train phonological AE
    Xae_wrds, deletedChars_wrds, oneLetter_wrds = XsSeg2XaePhon(Xs,
                                                      Xs_mask,
                                                      segs,
                                                      maxLen)

    Yae_wrds = getYae(Xae_wrds, reverseUtt, utts=False)

    ae_phon.fit(Xae_wrds,
                Yae_wrds,
                batch_size=batch_size,
                epochs=1)

    ## Train utterance AE
    Xae_utts, deletedChars_utts, oneLetter_utts = XsSeg2Xae(Xs,
                                                            Xs_mask,
                                                            segs,
                                                            maxUtt,
                                                            maxLen)
    Xae_emb = embed_words.predict(Xae_utts)
    Yae_emb = getYae(Xae_emb, reverseUtt, utts=False)


    ae_utt.fit(Xae_emb,
               Yae_emb,
               batch_size=batch_size,
               epochs=1)

    return Xae_utts, deletedChars_utts, oneLetter_utts

def updateSegmenter(segmenter, Xs, Xs_mask, segs, seg_shift, batch_size):
    charDim = Xs.shape[-1]
    maxChar = Xs_mask.shape[-1]
    seg_inputs = np.zeros((len(Xs), maxChar + seg_shift, charDim))
    seg_inputs[:, :maxChar, :] = Xs
    seg_mask = np.zeros((len(Xs_mask), maxChar + seg_shift))
    seg_mask[:, seg_shift:] = Xs_mask
    seg_mask = np.expand_dims(seg_mask, -1)
    seg_targets = np.zeros((len(segs), maxChar + seg_shift, 1))
    seg_targets[:, seg_shift:, :] = segs
    segHist = segmenter.fit([seg_inputs, seg_mask],
                            seg_targets,
                            batch_size=batch_size,
                            epochs=1)
    return segHist

def predictSegmenter(segmenter, Xs, Xs_mask, seg_shift, batch_size):
    charDim = Xs.shape[-1]
    maxChar = Xs_mask.shape[-1]
    seg_inputs = np.zeros((len(Xs), maxChar + seg_shift, charDim))
    seg_inputs[:, :maxChar, :] = Xs
    seg_mask = np.zeros((len(Xs_mask), maxChar + seg_shift))
    seg_mask[:, seg_shift:] = Xs_mask
    seg_mask = np.expand_dims(seg_mask, -1)
    return segmenter.predict([seg_inputs,seg_mask], batch_size=batch_size)[:, seg_shift:, :]

def trainAEOnly(ae_full, ae_phon, ae_utt, embed_words, Xs, Xs_mask, segs, maxUtt, maxLen, trainIters, batch_size, reverseUtt, acoustic):
    print('Training auto-encoder network')

    return updateAE(ae_phon,
                    ae_utt,
                    Xs,
                    Xs_mask,
                    segs,
                    maxUtt,
                    maxLen,
                    batch_size,
                    reverseUtt)[0]

def trainAEPhonOnly(ae, Xs, Xs_mask, segs, maxLen, trainIters, batch_size, reverseUtt):
    print('Training phonological auto-encoder network')

    ## Preprocess input data
    Xae, deletedChars, oneLetter = XsSeg2XaePhon(Xs,
                                                 Xs_mask,
                                                 segs,
                                                 maxLen)

    ## Randomly permute samples
    p, p_inv = getRandomPermutation(len(Xae))
    Xae = Xae[p]
    if reverseUtt:
        Yae = np.flip(Xae, 1)
    else:
        Yae = Xae

    ae.fit(Xae,
           Yae,
           batch_size=batch_size,
           epochs=trainIters)

    return Xae[p_inv]



def trainSegmenterOnly(segmenter, Xs, Xs_mask, Y, trainIters, batch_size, seg_shift):
    print('Training segmenter network')

    updateSegmenter(segmenter,
                    Xs,
                    Xs_mask,
                    Y,
                    seg_shift,
                    batch_size)

    print('Getting model predictions for evaluation')
    preds = predictSegmenter(segmenter,
                             Xs,
                             Xs_mask,
                             seg_shift,
                             batch_size)

    segsProposal = (preds > 0.5) * np.expand_dims(1-Xs_mask, -1)

    return segsProposal


def printSegAnalysis(model, Xs, Xs_mask, segs, maxUtt, maxLen, metric, batch_size, reverse_utt, acoustic):
    Xae_found, deletedChars_found, oneLetter_found = XsSeg2Xae(Xs,
                                                               Xs_mask,
                                                               segs,
                                                               maxUtt,
                                                               maxLen,
                                                               acoustic)

    Yae_found = getYae(Xae_found, reverse_utt)

    found_lossXutt = scoreXUtt(model,
                               Xae_found,
                               Yae_found,
                               batch_size,
                               reverse_utt,
                               metric)

    found_loss = found_lossXutt.sum()
    found_nChar = Xae_found.any(-1).sum()

    print('Total loss (sum of losses): %s' % found_loss)
    print('Deleted characters in segmentation: %d' % deletedChars_found.sum())
    print('Input characterrs in segmentation: %d' % found_nChar.sum())
    print('Loss per character: %.4f' % (float(found_loss) / found_nChar))
    print()

def plotPredsSeg(utt_ids, segmenter, Xs, Xs_mask, Y, logdir, prefix, iteration, seg_shift, batch_size, debug=False):
    ## Initialize plotting objects
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax_input = fig.add_subplot(311)
    ax_targ = fig.add_subplot(312)
    ax_pred = fig.add_subplot(313)

    inputs_raw = Xs[utt_ids]
    masks_raw = Xs_mask[utt_ids]
    preds_raw = predictSegmenter(segmenter,
                                 inputs_raw,
                                 masks_raw,
                                 seg_shift,
                                 batch_size)

    targs_raw = np.expand_dims(Y[utt_ids], -1)

    for u in range(len(utt_ids)):
        inputs = inputs_raw[u]
        inputs = inputs[np.where(1-masks_raw[u])]
        inputs = np.swapaxes(inputs, 0, 1)

        targs = targs_raw[u]
        targs = targs[np.where(1-masks_raw[u])]

        preds = preds_raw[u]
        preds = preds[np.where(1-masks_raw[u])]
        preds = np.squeeze(preds)

        ## Create and save plots
        fig.patch.set_visible(False)
        fig.suptitle('Utterance %d, Iteration %d' % (utt_ids[u], iteration))

        ax_input.clear()
        ax_input.axis('off')
        ax_input.set_title('Input', loc='left')
        hm_input = ax_input.pcolor(inputs, cmap=plt.cm.Blues)

        ax_targ.clear()
        ax_targ.axis('off')
        ax_targ.set_title('Target', loc='left')
        ax_targ.set_ylim([0,1])
        ax_targ.margins(0)
        hm_targ = ax_targ.bar(np.arange(len(targs)), targs)

        ax_pred.clear()
        ax_pred.axis('off')
        ax_pred.set_title('Prediction', loc='left')
        ax_pred.set_ylim([0,1])
        ax_pred.margins(0)
        hm_pred = ax_pred.bar(np.arange(len(preds)), preds)

        fig.savefig(logdir + '/barchart_' + prefix + '_utt' + str(utt_ids[u]) + '.jpg')


def plotPredsUtt(utt_ids, model, Xae, Yae, logdir, prefix, iteration, batch_size, debug=False):
    ## Initialize plotting objects
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax_input = fig.add_subplot(411)
    ax_targ = fig.add_subplot(412)
    ax_mean = fig.add_subplot(413)
    ax_pred = fig.add_subplot(414)
    inputs_raw = Xae[utt_ids]
    # preds_raw = model.predict(Xae[utt_ids], batch_size=batch_size)[0]
    preds_raw = model.predict(Xae[utt_ids], batch_size=batch_size)
    targs_raw = Yae[utt_ids]

    mean = np.expand_dims(np.mean(Yae, axis=(0, 1, 2)), -1)

    if debug:
        print('='*50)
        print('Segmentation details for 10 randomly-selected utterances')
    for u in range(len(utt_ids)):
        ## Remove word boundaries so reconstruction of entire utterance can be plotted
        inputs = []
        if debug:
            print('-'*50)
            print('Utterance %d' %(utt_ids[u]+1))
            sys.stdout.write('Input word lengths:')
        for w in range(len(inputs_raw[u])):
            inputs_w = inputs_raw[u,w,...]
            inputs_w = inputs_w[np.where(inputs_w.any(-1))]
            if debug:
                sys.stdout.write(' %d' %inputs_w.shape[0])
            inputs.append(inputs_w)
        inputs = np.concatenate(inputs)
        if debug:
            print('\nInput utt length: %d' %inputs.shape[0])
            sys.stdout.write('Prediction word lengths:')
        inputs = np.swapaxes(inputs, 0, 1)
        targs = []
        for w in range(len(targs_raw[u])):
            targs_w = targs_raw[u, w, ...]
            targs_w = targs_w[np.where(targs_w.any(-1))]
            if debug:
                sys.stdout.write(' %d' % targs_w.shape[0])
            targs.append(targs_w)
        targs = np.concatenate(targs)
        if debug:
            print('\nPrediction utt length: %d' % targs.shape[0])
            sys.stdout.write('Target word lengths:')
        targs = np.swapaxes(targs, 0, 1)
        preds = []
        for w in range(len(preds_raw[u])):
            preds_w = preds_raw[u,w,...]
            preds_w = preds_w[np.where(preds_w.any(-1))]
            if debug:
                sys.stdout.write(' %d' %preds_w.shape[0])
            preds.append(preds_w)
        preds = np.concatenate(preds)
        if debug:
            print('\nTarget utt length: %d' %preds.shape[0])
        preds = np.swapaxes(preds, 0, 1)

        ## Create and save plots
        fig.patch.set_visible(False)
        fig.suptitle('Utterance %d, Iteration %d' %(utt_ids[u], iteration))

        ax_input.clear()
        ax_input.axis('off')
        ax_input.set_title('Input', loc='left')
        hm_input = ax_input.pcolor(inputs, cmap=plt.cm.Blues)

        ax_targ.clear()
        ax_targ.axis('off')
        ax_targ.set_title('Target', loc='left')
        hm_targ = ax_targ.pcolor(targs, cmap=plt.cm.Blues)

        ax_mean.clear()
        ax_mean.axis('off')
        ax_mean.set_title('Target mean', loc='left')
        hm_mean = ax_mean.pcolor(mean, cmap=plt.cm.Blues)
        
        ax_pred.clear()
        ax_pred.axis('off')
        ax_pred.set_title('Prediction', loc='left')
        hm_pred = ax_pred.pcolor(preds, cmap=plt.cm.Blues)

        fig.savefig(logdir + '/heatmap_' + prefix + '_utt' + str(utt_ids[u]) + '.jpg')


def plotPredsWrd(wrd_ids, model, Xae, Yae, logdir, prefix, iteration, batch_size, debug=False):
    ## Initialize plotting objects
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax_input = fig.add_subplot(411)
    ax_targ = fig.add_subplot(412)
    ax_mean = fig.add_subplot(413)
    ax_pred = fig.add_subplot(414)
    inputs_raw = Xae[wrd_ids]
    preds_raw = model.predict(Xae[wrd_ids], batch_size=batch_size)
    targs_raw = Yae[wrd_ids]

    mean = np.expand_dims(np.mean(Yae, axis=(0, 1)), -1)

    if debug:
        print('=' * 50)
        print('Segmentation details for 10 randomly-selected utterances')
    for w in range(len(wrd_ids)):
        ## Remove word boundaries so reconstruction of entire utterance can be plotted
        if debug:
            print('-' * 50)
            print('Utterance %d' % (utt_ids[u] + 1))
            sys.stdout.write('Input word lengths:')
        inputs = inputs_raw[w, ...]
        inputs = inputs[np.where(inputs.any(-1))]
        if debug:
            sys.stdout.write(' %d' % inputs.shape[0])
        if debug:
            print('\nInput utt length: %d' % inputs.shape[0])
            sys.stdout.write('Prediction word lengths:')
        inputs = np.swapaxes(inputs, 0, 1)
        targs = targs_raw[w, ...]
        targs = targs[np.where(targs.any(-1))]
        if debug:
            sys.stdout.write(' %d' % targs.shape[0])
        if debug:
            print('\nPrediction utt length: %d' % targs.shape[0])
            sys.stdout.write('Target word lengths:')
        targs = np.swapaxes(targs, 0, 1)
        preds = preds_raw[w, ...]
        preds = preds[np.where(preds.any(-1))]
        if debug:
            sys.stdout.write(' %d' % preds.shape[0])
        if debug:
            print('\nTarget utt length: %d' % preds.shape[0])
        preds = np.swapaxes(preds, 0, 1)

        ## Create and save plots
        fig.patch.set_visible(False)
        fig.suptitle('Word %d, Iteration %d' % (wrd_ids[w], iteration))

        ax_input.clear()
        ax_input.axis('off')
        ax_input.set_title('Input', loc='left')
        hm_input = ax_input.pcolor(inputs, cmap=plt.cm.Blues)

        ax_targ.clear()
        ax_targ.axis('off')
        ax_targ.set_title('Target', loc='left')
        hm_targ = ax_targ.pcolor(targs, cmap=plt.cm.Blues)

        ax_mean.clear()
        ax_mean.axis('off')
        ax_mean.set_title('Target mean', loc='left')
        hm_mean = ax_mean.pcolor(mean, cmap=plt.cm.Blues)

        ax_pred.clear()
        ax_pred.axis('off')
        ax_pred.set_title('Prediction', loc='left')
        hm_pred = ax_pred.pcolor(preds, cmap=plt.cm.Blues)

        fig.savefig(logdir + '/heatmap_' + prefix + '_wrd' + str(utt_ids[w]) + '.jpg')


def runCheckpoint(Xs, Xs_mask, gold, otherParams, maxChar, maxLen, maxUtt, charDim, logdir, segShift, batchSize, reverseUtt, iteration, batch_num, acoustic=False, ae_full=None, segmenter=None, debug=False):
    if acoustic:
        intervals, GOLDWRD, GOLDPHN = otherParams
    else:
        ctable = otherParams
    ae_net = ae_full != None
    seg_net = segmenter != None
    print()
    print('Performing system evaluation')
    print('Segmenting cross-validation set')
    if seg_net:
        preds = predictSegmenter(segmenter,
                                 Xs_cv,
                                 Xs_mask_cv,
                                 segShift,
                                 batchSize)
        segs4eval = (preds > 0.5).astype(np.int8)
        if ACOUSTIC:
            segs4eval[np.where(vad_cv)] = 1.
        else:
            segs4eval[:, 0, ...] = 1.
        segs4eval[np.where(Xs_mask_cv)] = 0.
    if ae_net:
        Xae_cv, deletedChars_cv, oneLetter_cv = XsSeg2Xae(Xs_cv,
                                                          Xs_mask_cv,
                                                          segs4eval,
                                                          maxUtt,
                                                          maxLen,
                                                          acoustic)
        Yae_cv = getYae(Xae_cv, REVERSE_UTT)
        print('Computing network losses on cross-validation set')
        cvAELoss = lossXChar(ae_full, Xae_cv, Yae_cv, batchSize, acoustic, 'mse' if acoustic else 'xent')
        segPreds = preds
        cvSegLoss = -(
        np.nan_to_num(np.log(segPreds)) * segs4eval + np.nan_to_num(np.log(1 - segPreds)) * (1 - segs4eval))
        cvSegLoss *= np.expand_dims(1 - Xs_mask_cv, -1)
        cvSegLoss = cvSegLoss.sum() / (1 - Xs_mask_cv).sum()
        cvDel = deletedChars_cv.sum()
        cvOneL = oneLetter_cv.sum()
        cvSeg = segs4eval.sum()

        if not acoustic:
            print('')
            print('Example reconstruction of learned segmentation')
            printReconstruction(10, ae_full, Xae_cv, ctable, batchSize, reverseUtt)

        segs4evalXDoc = dict.fromkeys(doc_list_cv)
        for doc in segs4evalXDoc:
            s, e = doc_indices_cv[doc]
            segs4evalXDoc[doc] = segs4eval[s:e]
            if acoustic:
                masked_proposal = np.ma.array(segs4evalXDoc[doc], mask=Xs_mask_cv[s:e])
                segs4evalXDoc[doc] = masked_proposal.compressed()

        print('Scoring segmentation of cross-validation set')
        segScore = writeLog(batch_num,
                            iteration,
                            cvAELoss if ae_net else None,
                            None,
                            cvSegLoss if seg_net else None,
                            cvDel if ae_net else None,
                            cvOneL if ae_net else None,
                            cvSeg,
                            gold,
                            segs4evalXDoc,
                            logdir,
                            intervals if acoustic else None,
                            acoustic,
                            print_headers=not os.path.isfile(logdir + '/log.txt'))

        print('Total frames:', raw_total)
        if ae_net:
            print('Auto-encoder loss:', epochAELoss)
            print('Deletions:', epochDel)
            print('One letter words:', epochOneL)
        if SEG_NET:
            print('Segmenter loss:', epochSegLoss)
        print('Total segmentation points:', epochSeg)

        if acoustic:
            if GOLDWRD:
                print('Word segmentation scores:')
                printSegScores(segScore['wrd'], acoustic)
            if GOLDPHN:
                print('Phone segmentation scores:')
                printSegScores(segScore['phn'], acoustic)
            writeTimeSegs(frameSegs2timeSegs(intervals, segs4evalXDoc), out_dir=logdir, TextGrid=False)
            writeTimeSegs(frameSegs2timeSegs(intervals, segs4evalXDoc), out_dir=logdir, TextGrid=True)
        else:
            printSegScores(getSegScores(gold, segs4evalXDoc, acoustic), acoustic)
            writeSolutions(logdir, segs4evalXDoc[doc_list[0]], gold[doc_list[0]], batch_num)

        plotPredsUtt(utt_ids_cv,
                     ae_full,
                     Xae_cv,
                     Yae_cv,
                     logdir,
                     'train',
                     iteration,
                     batchSize,
                     debug)

        plotPredsSeg(utt_ids_cv,
                     segmenter,
                     Xs_cv,
                     Xs_mask_cv,
                     segs4eval,
                     logdir,
                     'train',
                     iteration + 1,
                     segShift,
                     batchSize,
                     debug)

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
    crossValDir = args.crossValDir if args.crossValDir else checkpoint.get('crossValDir', None)
    EVAL_FREQ = args.evalFreq if args.evalFreq else checkpoint.get('evalFreq', 50)
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
    SEG_SHIFT = args.segShift if args.segShift != None else checkpoint.get('segShift', 0 if ACOUSTIC else 0)
    pretrainIters = int(args.pretrainIters) if args.pretrainIters else checkpoint.get('pretrainIters', 10 if ACOUSTIC else 10)
    trainNoSegIters = int(args.trainNoSegIters) if args.trainNoSegIters else checkpoint.get('trainNoSegIters', 10 if ACOUSTIC else 10)
    trainIters = int(args.trainIters) if args.trainIters else checkpoint.get('trainIters', 100 if ACOUSTIC else 80)
    METRIC = args.metric if args.metric else checkpoint.get('metric', 'mse' if ACOUSTIC else 'logprob')
    DEL_WT = float(args.delWt) if args.delWt else checkpoint.get('delWt', 50 if ACOUSTIC else 50)
    ONE_LETTER_WT = float(args.oneLetterWt) if args.oneLetterWt else checkpoint.get('oneLetterWt', 50 if ACOUSTIC else 10)
    SEG_WT = float(args.segWt) if args.segWt else checkpoint.get('segWt', 0 if ACOUSTIC else 0)
    N_SAMPLES = int(args.nSamples) if args.nSamples else checkpoint.get('nSamples', 100 if ACOUSTIC else 100)
    BATCH_SIZE = checkpoint.get('batchSize', int(args.batchSize) if args.batchSize else 128 if ACOUSTIC else 128)
    SAMPLING_BATCH_SIZE = int(args.samplingBatchSize) if args.samplingBatchSize else checkpoint.get('samplingBatchSize', 128 if ACOUSTIC else 128)
    INITIAL_SEG_PROB = float(args.initialSegProb) if args.initialSegProb else checkpoint.get('initialSegProb', 0.2 if ACOUSTIC else 0.2)
    assert INITIAL_SEG_PROB >= 0 and INITIAL_SEG_PROB <= 1, 'Invalid value for initialSegProb (%.2f) -- must be between 0 and 1' %INITIAL_SEG_PROB
    INTERPOLATION_RATE = float(args.interpolationRate) if args.interpolationRate else checkpoint.get('interpolationRate', 0.1 if ACOUSTIC else 0.1)
    assert INTERPOLATION_RATE >= 0 and INTERPOLATION_RATE <= 1, 'Invalid value for interpolationRate (%.2f) -- must be between 0 and 1' %INTERPOLATION_RATE
    iteration = checkpoint.get('iteration', 0)
    pretrain = checkpoint.get('pretrain', True)
    DEBUG = args.debug
    RNN = recurrent.LSTM
    VIZ_COUNT = 10
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





    ##################################################################################
    ##################################################################################
    ##
    ##  Load training data
    ##
    ##################################################################################
    ##################################################################################

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


    ## Randomly pick utterance IDs for visualization
    utt_ids = np.random.choice(len(Xs), size=VIZ_COUNT, replace=False)
    if args.crossValDir:
        utt_ids_cv = np.random.choice(len(Xs_cv), size=VIZ_COUNT, replace=False)

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
            print('  Training data location: %s' % dataDir, file=f)
            print('  Training input type: Acoustic', file=f)
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
        print('  Search algorithm: %s' % ALGORITHM, file=f)
        print('  Optimizer: %s' % OPTIM, file=f)
        print('  Autoencoder loss function: %s' % METRIC, file=f)
        print('  Optimizer: %s' % OPTIM, file=f)
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
              '--crossValDir %s' if crossValDir else '',
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

            ## INPUT
            inp = Input(shape=(maxUtt, maxLen, charDim), name='AEfullInput')

            ## WORD ENCODER
            wordEncoder = Sequential(name='wordEncoder')
            wordEncoder.add(Dropout(charDropout,input_shape=(maxLen, charDim), noise_shape=(1, maxLen, 1)))
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
            wordDecoder.add(RepeatVector(maxLen, input_shape=(wordHidden,)))
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
            inp_phon = Input(shape=(maxLen, charDim), name='AEphonInput')
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
                                          REVERSE_UTT)

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
                                  ACOUSTIC)

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
                                      ACOUSTIC)

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
                                      ACOUSTIC)

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
                                                    Xs,
                                                    Xs_mask,
                                                    segs,
                                                    maxUtt,
                                                    maxLen,
                                                    BATCH_SIZE,
                                                    REVERSE_UTT)
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
            print('Training auto-encoder network on random segmentation.')
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
        epochSeg = checkpoint.get('epochSeg', 0)
        b =  checkpoint.get('b', 0)

        ## Randomly permute samples
        p, p_inv = getRandomPermutation(len(Xs))
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
                    pSegs_batch = pSegs_batch**2
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
                                                                               ACOUSTIC)

                    Yae_batch = getYae(Xae_batch, REVERSE_UTT)
                    input_batch = Xae_batch
                    target_batch = Yae_batch
                    scorerNetwork = ae_full
                else:
                    input_batch = Xs_batch
                    target_batch = segs_batch
                    scorerNetwork = segmenter

                scores_batch[:,s,:] = scoreXUtt(scorerNetwork,
                                                input_batch,
                                                target_batch,
                                                BATCH_SIZE,
                                                REVERSE_UTT,
                                                metric = METRIC)

                if AE_NET:
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
                        penalties_batch[u,s,-segs_batch[u].sum():] -= SEG_WT

            print('')

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

            st1 = time.time()
            print('Sampling time: %.2fs.' %(st1-st0))

            if AE_NET:
                Xae_batch, deletedChars_batch, oneLetter_batch = updateAE(ae_phon,
                                                                          ae_utt,
                                                                          Xs_batch,
                                                                          Xs_mask_batch,
                                                                          segsProposal_batch,
                                                                          maxUtt,
                                                                          maxLen,
                                                                          BATCH_SIZE,
                                                                          REVERSE_UTT)

                Yae_batch = getYae(Xae_batch, REVERSE_UTT)

                print('Fitting auto-encoder network')
                #model.load_weights(logdir + '/model_init.h5', by_name=True)
                aeHist = ae_full.fit(Xae_batch,
                                     Yae_batch,
                                     batch_size=BATCH_SIZE,
                                     epochs=1)
            if SEG_NET:
                print('Fitting segmenter network')
                segHist = updateSegmenter(segmenter,
                                          Xs_batch,
                                          Xs_mask_batch,
                                          segsProposal_batch,
                                          SEG_SHIFT,
                                          BATCH_SIZE)

                # print('Getting segmentation predictions from network')
                # segsProposal_batch = (segmenter.predict(Xs_batch) > 0.5).astype(np.int8)
                # if ACOUSTIC:
                #     segsProposal_batch[np.where(vad_batch)] = 1.
                # else:
                #     segsProposal_batch[:,0,...] = 1.
                # segsProposal_batch[np.where(Xs_mask_batch)] = 0.
                # n_Seg_batch = segsProposal_batch.sum()
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


            segsProposal.append(segsProposal_batch)
            epochSeg += int(n_Seg_batch)

            b += SAMPLING_BATCH_SIZE
            batch_num_global += 1

            if AE_NET:
                ae_full.save(logdir + '/model.h5')
            if SEG_NET:
                segmenter.save(logdir + '/segmenter.h5')
            with open(logdir + '/checkpoint.obj', 'wb') as f:
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
                checkpoint['epochSeg'] = epochSeg
                pickle.dump(checkpoint, f)

            ## Evaluate on cross-validation set
            if args.crossValDir and (batch_num_global % EVAL_FREQ == 0):
                if ACOUSTIC:
                    otherParams = intervals_cv, GOLDWRD_CV, GOLDPHN_CV
                else:
                    otherParams = ctable_cv
                runCheckpoint(Xs_cv,
                              Xs_mask_cv,
                              gold_cv,
                              otherParams,
                              maxChar,
                              maxLen,
                              maxUtt,
                              charDim,
                              logdir,
                              SEG_SHIFT,
                              BATCH_SIZE,
                              REVERSE_UTT,
                              iteration+1,
                              batch_num_global,
                              ACOUSTIC,
                              ae_full,
                              segmenter,
                              DEBUG)

            bt1 = time.time()
            print('Batch time: %.2fs' %(bt1-bt0))



        if AE_NET:
            epochAELoss /= N_BATCHES
            if not ACOUSTIC:
                epochAEAcc /= N_BATCHES
        if SEG_NET:
            epochSegLoss /= N_BATCHES
        segsProposal = np.concatenate(segsProposal)

        ## Invert random permutation so evaluation aligns correctly
        Xs = Xs[p_inv]
        Xs_mask = Xs_mask[p_inv]
        if ACOUSTIC:
            vad = vad[p_inv]
        pSegs = pSegs[p_inv]
        segsProposal = segsProposal[p_inv]

        iteration += 1
        b = 0

        if crossValDir == None:
            # Evaluate on training data
            if AE_NET:
                n = 10
                Xae, _, _ = XsSeg2Xae(Xs,
                                      Xs_mask,
                                      segsProposal,
                                      maxUtt,
                                      maxLen,
                                      ACOUSTIC)

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

            segsProposalXDoc = dict.fromkeys(doc_list)
            for doc in segsProposalXDoc:
                s,e = doc_indices[doc]
                segsProposalXDoc[doc] = segsProposal[s:e]
                if ACOUSTIC:
                    masked_proposal = np.ma.array(segsProposalXDoc[doc], mask=Xs_mask[s:e])
                    segsProposalXDoc[doc] = masked_proposal.compressed()

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
                                intervals if ACOUSTIC else None,
                                ACOUSTIC,
                                print_headers=not os.path.isfile(logdir + '/log.txt'))


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


        if AE_NET:
            ae_full.save(logdir + '/model.h5')
        if SEG_NET:
            segmenter.save(logdir + '/segmenter.h5')
        with open(logdir + '/checkpoint.obj', 'wb') as f:
            checkpoint['iteration'] = iteration
            checkpoint['batch_num_global'] = batch_num_global
            checkpoint['b'] = b
            checkpoint['segsProposal'] = []
            pickle.dump(checkpoint, f)

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
                                                                ACOUSTIC)

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
                                                                    ACOUSTIC)

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

