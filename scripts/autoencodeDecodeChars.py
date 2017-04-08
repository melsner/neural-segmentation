from __future__ import print_function, division
import argparse
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Model, Sequential, load_model
try:
    from keras.engine.training import slice_X
except:
    from keras.engine.training import _slice_arrays as slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Input, Reshape, Merge, merge, Lambda, Dropout
from keras import backend as K
import tensorflow as tf
import numpy as np
import cPickle as pickle
import random
import sys
import re
import copy
import time
from collections import defaultdict
from echo_words import CharacterTable, pad
from capacityStatistics import getPseudowords
from ae_io import *
from score import *

def uttsToCharVectors(text, maxchar, ctable):
    nUtts = len(text)
    X = np.zeros((nUtts, maxchar, ctable.dim()), dtype=np.bool)

    for ui, utt in enumerate(text):
        X[ui] = ctable.encode(pad(utt[:maxchar], maxchar, "X"), maxchar)

    return X

def uttsToFrameVectors(mfccs):
    uttlen = mfccs.shape[-2]
    maxlen = mfccs.shape[-1]
    new_shape = mfccs.shape[:-2]
    new_shape.append(uttlen*maxlen)
    return mfccs.reshape(new_shape)

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

def padFrameUtt(utt, maxutt, maxlen, FRAME_SIZE):
    deleted = sum([len(x) for x in utt[maxutt:]])
    one_lett = 0
    utt = utt[:maxutt]
    nWds = len(utt)
    for i,word in enumerate(utt):
        utt[i], deleted_wrd, one_lett_wrd = padFrameWord(word,maxlen,FRAME_SIZE)
        deleted += deleted_wrd
        one_lett += one_lett_wrd
    corr = max(0, maxutt - nWds)
    for i in xrange(corr):
        padWrd = np.zeros((maxlen,FRAME_SIZE))
        padWrd[:,-1] = 1.0
        utt.append(padWrd)
    utt = np.stack(utt, axis=0)
    assert utt.shape == (maxutt, maxlen, FRAME_SIZE), 'Utterance shape after padding should be %s, was actually %s.' %((maxutt, maxlen, FRAME_SIZE), utt.shape)
    return utt, deleted, one_lett

def padFrameWord(word, maxlen, FRAME_SIZE):
    deleted = max(0, word.shape[0]-maxlen)
    one_lett = int(word.shape[0] == 1)
    word = word[:maxlen,:]
    padChrs = np.zeros((max(0, maxlen-word.shape[0]),FRAME_SIZE))
    padChrs[:,-1] = 1.
    word = np.append(word, padChrs,0)
    assert word.shape == (maxlen,FRAME_SIZE), 'Word shape after padding should be %s, was actually %s.' %((maxlen,FRAME_SIZE), word.shape)
    return word, deleted, one_lett

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

def batchFrameIndices(forced, BATCH_SIZE, start_ix=0):
    end_ix = start_ix + BATCH_SIZE
    if end_ix >= len(forced):
        return start_ix, len(forced)
    for i in xrange(end_ix-1, start_ix-1, -1):
        end_ix = i
        if forced[i] == 1:
            break
    if end_ix > start_ix:
        return start_ix, end_ix
    return start_ix, start_ix + BATCH_SIZE

def charSegs2X(chars, segs, maxutt, maxlen, ctable):
    nUtts = len(chars)
    X = np.zeros((nUtts, maxutt, maxlen, ctable.dim()), dtype=np.bool)
    deletedChars = np.zeros((nUtts,))
    oneWord = np.zeros((nUtts,))

    for ui,uttChars in enumerate(chars):
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

def intervals2forcedSeg(intervals):
    out = dict.fromkeys(intervals)
    for doc in intervals:
        out[doc] = []
        offset = 0
        last = 0
        for i in intervals[doc]:
            s, e = i
            s, e = int(np.rint(s*100)), int(np.rint(e*100))
            interval = [0] * (e-s)
            interval[0] = 1
            offset += s-last
            out[doc] += interval
            last = e
        out[doc] = np.array(out[doc])
    return out

def segs2pSegs(segs, alpha=0.05):
    assert alpha >= 0.0 and alpha <= 0.5, 'Illegal value of alpha (0 <= alpha <= 0.5)'
    return np.maximum(alpha, segs - alpha)

def segs2pSegsWithForced(segs, forced, alpha=0.05):
    assert alpha >= 0.0 and alpha <= 0.5, 'Illegal value of alpha (0 <= alpha <= 0.5)'
    out = segs2pSegs(segs, alpha)
    out[np.where(forced)] = 1.
    return out

def sampleCharSegs(utts, pSegs):
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

def sampleFrameSegs(pSegs, intervals=None):
    smp = np.random.uniform(size=pSegs.shape)
    return smp < pSegs

def lossByUtt(model, Xb, yb, BATCH_SIZE, metric="logprob"):
    preds = model.predict(Xb, batch_size=BATCH_SIZE, verbose=0)

    if metric == "logprob":
        logP = np.log(preds)
        pRight = logP * yb
        #sum out word, char, len(chars)
        return pRight.sum(axis=(1, 2, 3))
    elif metric == 'mse':
        return np.mean((preds - yb)**2, axis=(1,2,3))
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

    if metric == "logprob":
        scores = np.array(scores)
        MM = np.max(scores, axis=0, keepdims=True)
        eScores = np.exp(scores - MM)
        #approximately the probability of the sample given the data
        pSeg = eScores / eScores.sum(axis=0, keepdims=True)
        utt_selector = np.arange(pSeg.shape[0])
        best = np.argmax(pSeg, axis=0)
        bestSegs=[]
        bestScore=0
        bestProb=0
        for ix in xrange(len(best)):
            bestSegs.append(segs[best[ix]][ix,:])
            bestScore += scores[best[ix],ix]
            bestProb += pSeg[best[ix],ix]
        bestProb /= len(best)
        bestSegs = np.array(bestSegs)
        print('      Best segmentation set: score = %s, prob = %s, num segs = %s' %(bestScore, bestProb, bestSegs.sum()))
        return bestSegs, bestSegs
    elif metric == 'mse':
        scores = np.array(scores)
        #scores = scores / (np.std(scores, axis=0)/10)
        MM = np.max(-scores, axis=0, keepdims=True)
        eScores = np.exp(-scores - MM)
        #approximately the probability of the sample given the data
        pSeg = eScores / eScores.sum(axis=0, keepdims=True)
        best = np.argmin(scores)
        print('      Best segmentation: ix = %s, score = %s, prob = %s, num segs = %s' %(best, scores[best][0], pSeg[best][0], segs[best].sum()))
        return segs[best], segs[best]
        seg_ix = np.where(np.random.multinomial(1, np.squeeze(pSeg)))[0][0]
        print('      Sampled segmentation: ix = %s, score = %s, prob = %s, num segs = %s' %(seg_ix, scores[seg_ix][0], pSeg[seg_ix][0], segs[seg_ix].sum()))
        print('')
        return segs[seg_ix], segs[seg_ix]
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

    #print("shape of segment matrix", segmat.shape)
    #print("losses for utt 0", scores[:, 0])
    #print("best score sample", np.argmax(-scores))
    #print("transformed losses for utt 0", eScores[:, 0])
    #print("best score sample", np.argmax(eScores))
    #best_score = np.argmax(eScores)

    #print("top row of distr", pSeg[:, 0])
    #print("top row of correction", qSeg[:, 0])
    #print("top row of weights", wts[:, 0])

    #sample x utterance x segment
    nSamples = segmat.shape[0]
    wtSegs = segmat * wts

    #for si in range(nSamples):
    #    print("seg vector", si, segmat[si, 0, :])
    #    print("est posterior", pSeg[si, 0])
    #    print("q", qSeg[si, 0])
    #    print("weight", wts[si, 0])
    #    print("contrib", wtSegs[si, 0])

    segWts = wtSegs.sum(axis=0)
    best = segWts > .5

    #print("difference between best sample and guessed best", np.sum(segs[best_score]-best))
    #print("total segmentations (best sample)", np.sum(segs[best_score]))
    #print("total segmentations (best guess)", np.sum(best))

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

def writeLog(iteration, epochLoss, epochDel, epochOneL, epochSeg, text, allBestSeg, logdir, intervals=None, acoustic=False, print_headers=False):
    if acoustic:
        allBestSeg = frameSegs2timeSegs(intervals,allBestSeg)
        headers = ['iteration', 'epochLoss', 'epochDel', 'epochOneL', 'epochSeg']
        scores_wrd = {}
        scores_phn = {}
        if text['wrd']:
            headers += ['bp_wrd', 'br_wrd', 'bf_wrd', 'swp_wrd', 'swr_wrd', 'swf_wrd']
            scores_wrd = getSegScore(text['wrd'],allBestSeg,acoustic=True)
        if text['phn']:
            headers += ['bp_phn', 'br_phn', 'bf_phn', 'swp_phn', 'swr_phn', 'swf_phn']
            scores_phn = getSegScore(text['phn'],allBestSeg,acoustic=True)
        for doc in set(scores_wrd.keys() + scores_phn.keys()):
            if not doc == '##overall##':
                score_row = [iteration, epochLoss, epochDel, epochOneL, epochSeg]
                if text['wrd'] and scores_wrd[doc]:
                    _, (bp,br,bf), _, (swp,swr,swf) = scores_wrd[doc]
                    score_row += [bp, br, bf, swp, swr, swf]
                if text['phn'] and scores_phn[doc]:
                    _, (bp,br,bf), _, (swp,swr,swf) = scores_phn[doc]
                    score_row += [bp, br, bf, swp, swr, swf]
                with open(logdir+doc+'_log.txt', 'ab') as f:
                    if print_headers:
                        print("\t".join(headers), file=f)
                    print("\t".join(["%g" % xx for xx in score_row]), file=f)
        with open(logdir+'log.txt', 'ab') as f:
            if print_headers:
                print("\t".join(headers), file=f)
            score_row = [iteration, epochLoss, epochDel, epochOneL, epochSeg]
            if text['wrd']:
                _, (bp,br,bf), _, (swp,swr,swf) = scores_wrd['##overall##']
                score_row += [bp, br, bf, swp, swr, swf]
            if text['phn']:
                _, (bp,br,bf), _, (swp,swr,swf) = scores_phn['##overall##']
                score_row += [bp, br, bf, swp, swr, swf]
            print("\t".join(["%g" % xx for xx in score_row]), file=f)
    else:
        allBestSeg = allBestSeg['main']
        text = text['main']
        segmented = matToSegs(allBestSeg, text)
        (bp,br,bf) = scoreBreaks(text, segmented)
        (swp,swr,swf) = scoreWords(text, segmented)
        (lp,lr,lf) = scoreLexicon(text, segmented)
        with open(logdir+'log.txt', 'ab') as f:
            if print_headers:
                print("\t".join([
                                "iteration", "epochLoss", "epochDel", 
                                "epochOneL", "epochSeg",
                                "bp", "br", "bf", "swp", "swr", "swf",
                                "lp", "lr", "lf"]), file=f)
            print("\t".join(["%g" % xx for xx in [
                            iteration, epochLoss, epochDel, epochOneL, epochSeg, bp, br, bf, swp, swr, swf,
                            lp, lr, lf]]), file=f)

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

    first = True
    FRAME_SIZE = 0

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
                        mfcc = map(float, line.strip().split()) + [0.0] # Extra dim for padding indicator
                    if first:
                        FRAME_SIZE = len(mfcc) 
                        first = False
                    assert len(mfcc) == FRAME_SIZE, 'ERROR: MFCC size (%d) in file "%s" does not agree with inferred frame size (%d).' %(len(mfcc)-1,p,FRAME_SIZE-1)
                    mfcc_lists[file_id].append(mfcc)
        mfcc_lists[file_id] = np.asarray(mfcc_lists[file_id])
    return mfcc_lists, FRAME_SIZE

def filterMFCCs(mfccs, intervals, segs, FRAME_SIZE=40):
    # Filter out non-speech portions
    mfcc_intervals = {}
    for doc in segs:
        mfcc_intervals[doc] = np.zeros((0,FRAME_SIZE))
        for i in intervals[doc]:
            sf, ef = int(np.rint(float(i[0]*100))), int(np.rint(float(i[1]*100)))
            mfcc_intervals[doc] = np.append(mfcc_intervals[doc], mfccs[doc][sf:ef,:], 0)
        print('Document "%s" has %d speech frames.' %(doc, len(mfccs[doc])))
    return mfcc_intervals


def splitMFCCs(mfcc_intervals,segs,maxutt,maxlen,maxchar,FRAME_SIZE,start=0,n_utt=np.inf,word_segs=True):
    # Split mfcc intervals according to segs
    end = start
    out = {}
    deletedChars = []
    one_lett = []
    words = np.split(mfcc_intervals[start:], np.where(segs[start:])[0])[1:]
    utts = []
    utt = []
    utt_len_chars = 0
    w = words.pop(0)
    w_len = w.shape[0]
    end += w_len
    while w != None and len(utt) < n_utt:
        # Handle case when single word length exceeds maxchar
        if w_len > maxchar:
            utt.append(w)
            if len(words) > 0:
                w = words.pop(0)
                w_len = w.shape[0]
                end += w_len
            else:
                w = None
        else:
            while utt_len_chars + w_len <= maxchar:
                utt.append(w)
                utt_len_chars += w_len
                if len(words) > 0:
                    w = words.pop(0)
                    w_len = w.shape[0]
                    end += w_len
                else:
                    w = None
                    break
        if word_segs:
            utt, deleted_utt, one_lett_utt = padFrameUtt(utt, maxutt, maxlen, FRAME_SIZE)
        else:
            utt, deleted_utt, one_lett_utt = padFrameWord(np.concatenate(utt,axis=0), maxchar, FRAME_SIZE)
        utts.append(utt)
        deletedChars.append(deleted_utt)
        one_lett.append(one_lett_utt)
        if w != None:
            utt = []
            utt_len_chars = 0
        else:
            utt = None
    if utt != None:
        utts.append(utt)
        deletedChars.append(deleted_utt)
        one_lett.append(one_lett_utt)
    utts = np.stack(utts,axis=0)
    deletedChars = np.asarray(deletedChars)
    one_lett = np.asarray(one_lett)
    return utts, deletedChars, one_lett, end

def concatDocs(docs):
    doc_list = [docs[x] for x in sorted(docs.keys())]
    return np.concatenate(doc_list, axis=0)

def timeSeg2frameSeg(timeseg_file):
    intervals = {}
    speech = {}
    offsets = {}
    seg = 0
    with open(timeseg_file, 'rb') as f:
        for line in f:
            if line.strip() != '':
                doc, start, end = line.strip().split()[:3]
                if doc in intervals:
                    if float(start) <= intervals[doc][-1][1]:
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
                    # In rare cases, the Rasanen pre-seg system
                    # generates a start time earlier than previous
                    # interval's end time, requiring us to check this.
                    s = max(s,speech[doc][-1][1])
                    speech[doc].append((s-offset,e-offset,seg))
                else:
                    speech[doc] = [(s-offset,e-offset,seg)]

    segs = {}
    for doc in speech:
        segs[doc] = np.zeros((speech[doc][-1][1]))
        for seg in speech[doc]:
            segs[doc][seg[0]] = 1.0

    return intervals, segs 

def frameSegs2timeSegs(intervals, seg_f):
    out = dict.fromkeys(intervals)
    for doc in intervals:
        out[doc] = frameSeg2timeSeg(intervals[doc],seg_f[doc])
    return out

def frameSeg2timeSeg(intervals, seg_f):
    offset = last_interval = last_seg = 0
    this_frame = 0
    next_frame = 1
    seg_t = []
    for i in intervals:
        # Interval boundaries in seconds (time)
        st, et = i
        # Interval boundaries in frames
        sf, ef = int(np.rint(float(st)*100)), int(np.rint(float(et)*100))

        offset += sf - last_interval
        last_interval = ef
        while this_frame + offset < ef:
            if next_frame >= seg_f.shape[0] or np.allclose(seg_f[next_frame], 1):
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

def printTimeSeg(seg, out_file=sys.stdout, docname=None, TextGrid=False):
    if TextGrid:
        print('File type = "ooTextFile"', file=out_file)
        print('Object class = "TextGrid"', file=out_file)
        print('', file=out_file)
        print('xmin = %.3f' %seg[0][0], file=out_file)
        print('xmax = %.3f' %seg[-1][1], file=out_file)
        print('tiers? <exists>', file=out_file)
        print('size = 1', file=out_file)
        print('item []:', file=out_file)
        print('    item [1]:', file=out_file)
        print('        class = "IntervalTier"', file=out_file)
        print('        class = "segmentations"', file=out_file)
        print('        xmin = %.3f' %seg[0][0], file=out_file)
        print('        xmax = %.3f' %seg[-1][1], file=out_file)
        print('        intervals: size = %d' %len(seg), file=out_file)
        i = 1
        for s in seg:
            print('        intervals [%d]:' %i, file=out_file)
            print('            xmin = %.3f' %s[0], file=out_file)
            print('            xmax = %.3f' %s[1], file=out_file)
            print('            text = ""', file=out_file)
            i += 1
        print('', file=out_file)

    else:
        for i in seg:
            if docname:
                print('%s %s %s' %(docname, i[0], i[1]), file=out_file)
            else:
                print('%s %s' %i, file=out_file)

def printTimeSegs(segs, out_file='./', TextGrid=False):
    assert type(out_file) == str, 'out_file must be a directory path.'
    out_path = out_file
    if TextGrid:
        suffix = '.TextGrid'
    else:
        suffix = '_seg.txt'
    for doc in segs:
        with open(out_path + doc + suffix, 'wb') as f:
            printTimeSeg(segs[doc], out_file=f, docname=doc, TextGrid=TextGrid)

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
            
def KL(pSeg1,pSeg2):
    return np.mean(pSeg1*(np.log(pSeg1/pSeg2)) + (1-pSeg1)*(np.log((1-pSeg1)/(1-pSeg2))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    #parser.add_argument("pseudWeights")
    parser.add_argument("--uttHidden", default=400)
    parser.add_argument("--wordHidden", default=40)
    parser.add_argument("--segHidden", default=100)
    parser.add_argument("--wordDropout", default=.5)
    parser.add_argument("--charDropout", default=.5)
    parser.add_argument("--pretrainIters", default=None)
    parser.add_argument("--trainNoSegIters", default=None)
    parser.add_argument("--trainIters", default=None)
    parser.add_argument("--maxChar", default=None)
    parser.add_argument("--maxLen", default=None)
    parser.add_argument("--maxUtt", default=None)
    parser.add_argument("--nSamples", default=None)
    parser.add_argument("--batchSize", default=None)
    parser.add_argument("--logfile", default=None)
    parser.add_argument("--acoustic", action='store_true')
    parser.add_argument("--segfile", default=None)
    parser.add_argument("--goldwrd", default=None)
    parser.add_argument("--goldphn", default=None)
    parser.add_argument("--gpufrac", default=0.15)
    args = parser.parse_args()
    try:
        args.gpufrac = float(args.gpufrac)
    except:
        args.gpufrac = None

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpufrac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)
    K.set_learning_phase(1)
    print(K.learning_phase())

    if args.acoustic:
        assert args.segfile and (args.goldwrd or args.goldphn), 'Files containing initial and gold segmentations are required in acoustic mode.'
        if not args.pretrainIters:
            args.pretrainIters = 100
        if not args.trainNoSegIters:
            args.trainNoSegIters = 10
        if not args.trainIters:
            args.trainIters = 101
        if not args.nSamples:
            args.nSamples = 100
        if not args.batchSize:
            args.batchSize = 1000
        if not args.maxLen:
            args.maxLen = 100
        if not args.maxUtt:
            args.maxUtt = 50
        if not args.maxChar:
            args.maxChar = 400
    else:
        if not args.pretrainIters:
            args.pretrainIters = 10
        if not args.trainNoSegIters:
            args.trainNoSegIters = 10
        if not args.trainIters:
            args.trainIters = 81
        if not args.nSamples:
            args.nSamples = 50
        if not args.batchSize:
            args.batchSize = 128
        if not args.maxLen:
            args.maxLen = 7
        if not args.maxUtt:
            args.maxUtt = 10
        if not args.maxChar:
            args.maxChar = 30

    path = args.data
    #pseudWeights = args.pseudWeights

    # path = sys.argv[1]
    # pseudWeights = sys.argv[2] #"pseud-echo-weights.h5"

    t0 = time.time()
    print()
    print('Loading data...')
    print()

    if args.acoustic:
        # intervals, segs_init, and mfccs are dictionaries
        # indexed by document ID
        intervals, segs_init = timeSeg2frameSeg(args.segfile)
        forced = intervals2forcedSeg(intervals)
        text = {'wrd': None, 'phn': None}
        if args.goldwrd:
            text['wrd'] = readGoldFrameSeg(args.goldwrd)
        if args.goldphn:
            text['phn'] = readGoldFrameSeg(args.goldphn)
        mfccs, FRAME_SIZE = readMFCCs(path)
        mfccs = filterMFCCs(mfccs, intervals, segs_init, FRAME_SIZE)
        doc_list = sorted(list(mfccs.keys()))
        DEL_WT = 1 
        ONE_LETTER_WT = 50
        SEG_PENALTY = 0

        if text['wrd']:
            print('Initial word segmentation scores:')
            printSegScore(getSegScore(text['wrd'], frameSegs2timeSegs(intervals,segs_init), args.acoustic),True)
        if text['phn']:
            print('Initial phone segmentation scores:')
            printSegScore(getSegScore(text['phn'], frameSegs2timeSegs(intervals,segs_init), args.acoustic),True)
        print()
    else:
        textGold, uttChars, charset = readText(path)
        text = {'main': textGold}
        print('corpus length:', len(text))
        chars = ["X"] + charset
        print('total chars:', len(chars))
        ctable = CharacterTable(chars)
        intervals = None # Speech intervals not relevant for character mode
        ## TODO: Change symbolic mode to allow multiple input files
        ## like acoustic mode currently does
        doc_list = ['main']
        DEL_WT = 50
        ONE_LETTER_WT = 50
    
    t1 = time.time()
    print('Data loaded in %ds.' %(t1-t0))
    print()
    
    hidden = int(args.wordHidden) #40
    wordDecLayers = 1
    uttHidden = int(args.uttHidden) #400
    segHidden = int(args.segHidden) #100
    wordDropout = float(args.wordDropout) #.5
    charDropout = float(args.charDropout) #.5
    maxlen = int(args.maxLen)
    maxutt = int(args.maxUtt)
    maxchar = int(args.maxChar)
    N_SAMPLES = int(args.nSamples)
    BATCH_SIZE = int(args.batchSize)
    pretrain_iters = int(args.pretrainIters)
    train_noseg_iters = int(args.trainNoSegIters)
    train_tot_iters = int(args.trainIters)
    RNN = recurrent.LSTM
    reverseUtt = True
    if args.acoustic:
        METRIC = 'mse'
    else:
        METRIC = "logprob"
    charDim = FRAME_SIZE if args.acoustic else len(chars)

    wordEncoder = Sequential()
    wordEncoder.add(Dropout(charDropout, input_shape=(maxlen, charDim),
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

    wordDecoder.add(TimeDistributed(Dense(charDim, name="dense"), name="td"))
    if args.acoustic:
        wordDecoder.add(Activation('linear', name='linear'))
    else:
        wordDecoder.add(Activation('softmax', name="softmax"))

    #wordEncoder.load_weights(pseudWeights, by_name=True)
    #wordDecoder.load_weights(pseudWeights, by_name=True)

    print("Build full model...")

    #input word encoders
    inp = Input(shape=(maxutt, maxlen, charDim))
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
    if args.acoustic:
        model.compile(loss='mean_squared_logarithmic_error',
                      optimizer='adam',
                      metrics=['cosine_proximity'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    model.summary()

    seg_len = BATCH_SIZE if args.acoustic else maxchar

    segmenter = Sequential()
    segmenter.add(RNN(segHidden, input_shape=(seg_len, charDim),
                      return_sequences=True, name="segmenter"))
    segmenter.add(TimeDistributed(Dense(1)))
    segmenter.add(Activation("sigmoid"))
    segmenter.compile(loss="binary_crossentropy",
                      optimizer="adam")
    segmenter.summary()

    # Set up logging, load any saved data
    load_models = False
    if args.logfile == None:
        logdir = "logs/" + str(os.getpid()) + '/'
    else:
        logdir = "logs/" + args.logfile

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        load_models = True
    
    print("Logging at", logdir)
    
    if load_models and os.path.exists(logdir + 'model.h5'):
        print('Autoencoder checkpoint found. Loading weights...')
        model.load_weights(logdir + 'model.h5', by_name=True)
    else:
        print('No autoencoder checkpoint found. Keeping default initialization.')
    if load_models and os.path.exists(logdir + 'segmenter.h5'):
        print('Segmenter checkpoint found. Loading weights...')
        segmenter.load_weights(logdir + 'segmenter.h5', by_name=True)
    else:
        print('No segmenter checkpoint found. Keeping default initialization.')
    if load_models and os.path.exists(logdir + 'checkpoint.obj'):
        print('Training checkpoint data found. Loading...')
        with open(logdir + 'checkpoint.obj', 'rb') as f:
            obj = pickle.load(f)
        iteration = obj['iteration']
        pretrain = obj['pretrain']
        random_doc_list = obj['random_doc_list']
        doc_ix = obj['doc_ix']
        allBestSegs = obj.get('allBestSegs', None)
        deletedChars = obj['deletedChars']
        oneLetter = obj['oneLetter']
        reshuffle_doc_list=False
    else:
        print('No training checkpoint found. Starting training from beginning.')
        iteration = 0
        pretrain = True
        allBestSegs = None
        deletedChars = None
        oneLetter = None
        reshuffle_doc_list=True

    print()
    print("Pre-training autoencoder...")

    if args.acoustic:
        mfccs_joint = concatDocs(mfccs)
        segs_init_joint = concatDocs(segs_init)

    X_train = dict.fromkeys(doc_list)
    utt_segs = dict.fromkeys(doc_list)
    utt_lens = dict.fromkeys(doc_list)
    if deletedChars == None:
        deletedChars = dict.fromkeys(doc_list)
    if oneLetter == None:
        oneLetter = dict.fromkeys(doc_list)

    ## pretrain
    while iteration < pretrain_iters and pretrain:
        print(iteration)
        print()
        print('-' * 50)
        print('Iteration', iteration)

        if reshuffle_doc_list:
            random_doc_list = doc_list[:]
            random.shuffle(random_doc_list)
            doc_ix=0
        while doc_ix < len(random_doc_list):
            doc = random_doc_list[doc_ix]

            #seg_lengths = []
            #for i in range(1,len(forced[doc])-1):
            #    seg_lengths.append(forced[doc][i]-forced[doc][i-1])
            #seg_lengths = np.array(seg_lengths)
            #print(np.percentile(seg_lengths,50))

            if args.acoustic:
                print(doc)
                pSegs = segs2pSegsWithForced(segs_init[doc], forced[doc], alpha = 0.05)
                segs = sampleFrameSegs(pSegs)
                X_train[doc],deletedChars[doc],oneLetter[doc],chunk_length = splitMFCCs(mfccs[doc], segs, maxutt, maxlen, maxchar, FRAME_SIZE)
            else:
                utts = uttChars
                pSegs = .2 * np.ones((len(utts), maxchar))
                segs = sampleCharSegs(utts, pSegs)
                X_train[doc],deletedChars[doc],oneLetter = charSegs2X(utts, segs,
                                                     maxutt, maxlen, ctable)
            if reverseUtt:
                y_train = X_train[doc][:, ::-1, :]
            else:
                y_train = X_train[doc]

            print("Actual deleted chars from document %s:" %doc, deletedChars[doc].sum())
            model.fit(X_train[doc], y_train, batch_size=BATCH_SIZE, nb_epoch=1)

            doc_ix += 1
            
            model.save(logdir + 'model.h5')
            with open(logdir + 'checkpoint.obj', 'wb') as f:
                obj = {'iteration': iteration,
                       'pretrain': True,
                       'random_doc_list': random_doc_list,
                       'doc_ix': doc_ix,
                       'deletedChars': deletedChars,
                       'oneLetter': oneLetter}
                pickle.dump(obj, f)

            toPrint = 10
            preds = model.predict(X_train[doc][:toPrint], verbose=0)
            if reverseUtt:
                preds = preds[:, ::-1, :]

            for utt in range(toPrint):
                if args.acoustic:
                    pass
                    #print('Source (1st dim, 1st 100 characters):')
                    #print(mfccs[doc][:100,0])
                    #for i in range(5):
                    #    print('Segmented (1st dim, word %i):' %i)
                    #    print(X_train[doc][0,i,:,0])
                    #print()
                else:
                    thisSeg = segs[utt]
                    rText = reconstruct(utts[utt], thisSeg, maxutt)

                    print(realize(rText, maxlen, maxutt))

                    for wi in range(maxutt):
                        guess = ctable.decode(preds[utt, wi], calc_argmax=True)
                        print(guess, end=" ")
                    print("\n")


        iteration += 1
        reshuffle_doc_list=True
        if iteration == pretrain_iters:
            pretrain = False
            iteration = 0

    if allBestSegs == None:
        allBestSegs = dict.fromkeys(doc_list)
        for doc in doc_list:
            if args.acoustic:
                allBestSegs[doc] = np.zeros(segs_init[doc].shape)
            else:
                allBestSegs[doc] = np.zeros((X_train[doc].shape[0], maxchar))
    if args.acoustic:
        segs = segs_init
        XC = mfccs
    else:
        XC = {'main': uttsToCharVectors(uttChars, maxchar, ctable)}
    print()

    print('Co-training autoencoder and segmenter...')
    while iteration < train_tot_iters:
        t0 = time.time()
        print()
        print('-' * 50)
        print('Iteration', iteration)
        if iteration < train_noseg_iters:
            print(' Using initialization as proposal probability.')
        else:
            print(' Using segmenter network output as proposal probability.')

        ## mixed on/off policy learning?
        # alpha = min(1, (iteration / 30.))

        epochLoss = 0
        epochDel = 0
        epochOneL = 0
        epochSeg = 0

        tdoc1 = None
        tdoc2 = None

        if reshuffle_doc_list:
            random_doc_list = doc_list[:]
            random.shuffle(random_doc_list)
            doc_ix=0
        while doc_ix < len(random_doc_list):
            doc = random_doc_list[doc_ix]
            tdoc2 = time.time()
            if tdoc1 != None:
                print('  Completed in %ds.' %(tdoc2-tdoc1))
            print('Processing file %d/%d: "%s".' %(doc_ix+1,len(doc_list),doc))
            sys.stdout.flush()
            tdoc1 = time.time()
            if args.acoustic: # Don't batch by utt
                s = 0
                batch_ix = 0
                while s < XC[doc].shape[0]:
                    batch_ix += 1
                    print('  Batch %d. Starting at frame %d.' %(batch_ix,s))
                    s, e = batchFrameIndices(forced[doc], BATCH_SIZE, start_ix=s)
                    printSome = False
                    batch_in, _, _ = padFrameWord(XC[doc][s:e], BATCH_SIZE, FRAME_SIZE)
                    if iteration < train_noseg_iters:
                        nSamples = N_SAMPLES
                        pSegs = segs2pSegsWithForced(segs_init[doc], forced[doc], alpha = 0.05)
                        pSegs = np.resize(pSegs[s:e], BATCH_SIZE)
                    else:
                        nSamples = N_SAMPLES
                        pSegs = segmenter.predict(np.expand_dims(batch_in,0), verbose=0)
                        pSegs = np.squeeze(pSegs)
                        pSegs[np.where(forced[doc][s:e])] = 1.
                    scores = []
                    segSamples = []
                    print('    Sampling and scoring segmentations.')
                    for sample in range(nSamples):
                        segs = np.resize(sampleFrameSegs(pSegs), BATCH_SIZE)
                        X,deleted,oneLetter,batch_len_char = splitMFCCs(batch_in, segs, maxutt, maxlen, maxchar, FRAME_SIZE)
                        if reverseUtt:
                            y = X[:, ::-1, :]
                        else:
                            y = X

                        loss = lossByUtt(model, X, y, X.shape[0], metric=METRIC)

                        #print(loss.sum())
                        #print((DEL_WT * deleted).sum())
                        #print((ONE_LETTER_WT * oneLetter).sum())
                        #print('')
                        #print((np.sum(loss + \
                        #                     DEL_WT * deleted + \
                        #                     ONE_LETTER_WT * oneLetter) + \
                        #                     SEG_PENALTY * segs.sum())[None,...])
                        #scores.append(np.ones((1)))
                        scores.append((np.sum(loss + \
                                             DEL_WT * deleted + \
                                             ONE_LETTER_WT * oneLetter) + \
                                             SEG_PENALTY * segs.sum())[None,...])
                        segSamples.append(np.resize(segs, (1, BATCH_SIZE)))
                    segProbs, bestSegs = guessSegTargets(scores, segSamples, pSegs[None,...],
                                                         metric=METRIC)
                    bestSegs[0,np.where(forced[doc][s:e])] = 1.
                   
                    # Testing random baseline
                    #p = (np.zeros(bestSegs.shape) + 0.1)
                    #p[0, np.where(forced[doc][s:e])] = 1.
                    #bestSegs = sampleFrameSegs(p)

                    X,deleted,oneLetter,batch_len_char = splitMFCCs(batch_in, np.squeeze(bestSegs),
                                                        maxutt, maxlen, maxchar, FRAME_SIZE)
                    if reverseUtt:
                        y = X[:, ::-1, :]
                    else:
                        y = X

                    if s == 0:
                        allBestSegs[doc] = np.squeeze(bestSegs)[:e-s]
                    else:
                        allBestSegs[doc] = np.append(allBestSegs[doc],np.squeeze(bestSegs)[:e-s])

                    print('    Updating models.')
                    loss = model.train_on_batch(X, y)
                    segmenter.train_on_batch(np.expand_dims(batch_in,0), np.expand_dims(segProbs, 2))
                    seg_out = np.squeeze(segmenter.predict(np.expand_dims(batch_in,0), verbose=0))
                    #print("      Peakiness in segmenter output (mean KL divergence from uniform):", KL(seg_out, np.ones(seg_out.shape)/2))
                    print("      Number of probs > .5 in segmenter output:", (seg_out > 0.5).sum())
                    print("      Number of probs > .2 in segmenter output:", (seg_out > 0.2).sum())
                    print("      Number of probs > .15 in segmenter output:", (seg_out > 0.15).sum())
                    print("      Number of probs > .1 in segmenter output:", (seg_out > 0.1).sum())
                    #print("      First 10 segProbs:", segProbs[0,:10])
                    #print("      First 10 bestSegs:", bestSegs[0,:10])
                    #print("      First 10 segmenter probs:", segmenter_out[:10])
                    epochLoss += loss[0]
                    epochDel += deleted.sum()
                    epochOneL += oneLetter.sum()
                    epochSeg += bestSegs[:e-s].sum()

                    s = e
                epochLoss /= batch_ix

            else:
                for batch, inds in enumerate(batchIndices(X_train[doc], BATCH_SIZE)):
                    printSome = False
                    if batch % 25 == 0:
                        print("Batch:", batch)
                        if batch == 0:
                            printSome = True

                    XCb = XC[doc][inds]
                    utts = uttChars[inds]
                    segLen = len(utts)

                    if iteration < 10:
                        nSamples = N_SAMPLES
                        pSegs = .1 * np.ones((segLen, maxchar))
                    else:
                        nSamples = N_SAMPLES
                        pSegs = segmenter.predict(XCb, verbose=0)
                        #original shape has trailing 1
                        pSegs = np.squeeze(pSegs, -1)

                        #smooth this out a bit?
                        pSegs = .9 * pSegs + .1 * .5 * np.ones((segLen, maxchar))
                        #print("pseg shape", pSegs.shape)

                    ## interpolate policies
                    # pSegsOff = .05 * np.ones((segLen, maxchar))
                    # pSegsOn = segmenter.predict(XCb, verbose=0)
                    # pSegsOn = np.squeeze(pSegsOn, -1)
                    # pSegs = (1 - alpha) * pSegsOff + alpha * pSegsOn

                    scores = []
                    segSamples = []
                    for sample in range(nSamples):
                        segs = sampleCharSegs(utts, pSegs)
                        Xb,deletedChars,oneLetter = charSegs2X(utts, segs,
                                                                maxutt, maxlen, ctable)
                        if reverseUtt:
                            yb = Xb[:, ::-1, :]
                        else:
                            yb = Xb

                        loss = lossByUtt(model, Xb, yb, BATCH_SIZE, metric=METRIC)
                        scores.append(loss - DEL_WT * deletedChars
                                      - ONE_LETTER_WT * oneLetter)
                        segSamples.append(segs)

                    segProbs, bestSegs = guessSegTargets(scores, segSamples, pSegs,
                                                         metric=METRIC)
                   
                    Xb, deleted, oneLetter = charSegs2X(utts, bestSegs,
                                                     maxutt, maxlen, ctable)
                    if reverseUtt:
                        yb = Xb[:, ::-1, :]
                    else:
                        yb = Xb

                    allBestSegs[doc][inds] = bestSegs

                    loss = model.train_on_batch(Xb, yb)
                    segmenter.train_on_batch(XCb, np.expand_dims(segProbs, 2))
                    epochLoss += loss[0]
                    epochDel += deleted.sum()
                    epochOneL += oneLetter.sum()
                    epochSeg += bestSegs.sum()

                    # if batch % 25 == 0:
                    #     print("Loss:", loss)
                    #     print("Mean deletions:", np.array(dels[doc]).sum(axis=1).mean())
                    #     print("Deletions in best:", deleted.sum())

                    if printSome:
                        toPrint = 10

                        predLst = []

                        for smp in range(nSamples):
                            segs = segSamples[smp]
                            Xb, deleted,oneLetter = charSegs2X(utts, segs,
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
                        Xb, deleted, oneLetter = charSegs2X(utts, segs,
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
                                if args.acoustic:
                                    pass
                                else:
                                    #print(utts[utt])

                                    thisSeg = segSamples[smp][utt]
                                    rText = reconstruct(utts[utt], thisSeg, maxutt)

                                    print(realize(rText, maxlen, maxutt))

                                    for wi in range(maxutt):
                                        guess = ctable.decode(
                                            predLst[smp][utt, wi], calc_argmax=True)
                                        print(guess, end=" ")
                                    print()
                                    print([smp])
                                    print([utt])
                                    print("Score", scores[smp][utt], "del", deleted[utt])
                        print()

            doc_ix += 1
                
            model.save(logdir + 'model.h5')
            segmenter.save(logdir + 'segmenter.h5')
            with open(logdir + 'checkpoint.obj', 'wb') as f:
                obj = {'iteration': iteration,
                       'pretrain': False,
                       'random_doc_list': random_doc_list,
                       'doc_ix': doc_ix,
                       'allBestSegs': allBestSegs,
                       'deletedChars': deletedChars,
                       'oneLetter': oneLetter}
                pickle.dump(obj, f)


        tdoc2 = time.time()
        if tdoc1 != None:
            print(' Completed in %ds.' %(tdoc2-tdoc1))

        t1 = time.time()
        print("Iteration total time: %ds" %(t1-t0))
        print("Loss:", epochLoss)
        print("Deletions:", epochDel)
        print("One letter words:", epochOneL)
        print("Total segmentation points:", epochSeg)
        if args.acoustic:
            if text['wrd']:
                print('Word segmentation scores:')
                printSegScore(getSegScore(text['wrd'], frameSegs2timeSegs(intervals,allBestSegs), args.acoustic),args.acoustic)
            if text['phn']:
                print('Phone segmentation scores:')
                printSegScore(getSegScore(text['phn'], frameSegs2timeSegs(intervals,allBestSegs), args.acoustic),args.acoustic)
            printTimeSegs(frameSegs2timeSegs(intervals,allBestSegs), out_file=logdir, TextGrid=False) 
            printTimeSegs(frameSegs2timeSegs(intervals,allBestSegs), out_file=logdir, TextGrid=True) 

        else:
            printSegScore(getSegScore(text, allBestSegs, args.acoustic),args.acoustic)
        writeLog(iteration, epochLoss, epochDel, epochOneL, epochSeg, 
                 text, allBestSegs, logdir, intervals, args.acoustic, print_headers=iteration==0)

        if iteration % 10 == 0:
            if args.acoustic:
                printTimeSegs(frameSegs2timeSegs(intervals,allBestSegs), out_file=logdir) 
            else:
                writeSolutions(logdir, model, segmenter,
                               allBestSegs['main'], text['main'], iteration)
        doc_ix = 0
        iteration += 1

    print("Logs in", logdir)

