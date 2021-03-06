from __future__ import print_function, division
import sys, re, numpy as np
from scipy.signal import resample, argrelmax
from numpy import inf
from echo_words import CharacterTable, pad





##################################################################################
##################################################################################
##
##  GENERAL PURPOSE METHODS
##
##################################################################################
##################################################################################

def segs2pSegs(segs, interpolationRate=0.1):
    return (1-interpolationRate) * segs + interpolationRate * 0.5 * np.ones_like(segs)

def sampleSeg(pSegs, acoustic=False, resamplePSegs=False, concentration=10):
    if resamplePSegs:
        smp = np.zeros_like(pSegs)
        for i in range(len(pSegs)):
            for j in range(len(pSegs[i])):
                if pSegs[i,j,0] > 0 and pSegs[i,j,0] < 1:
                    smp[i,j,0] = np.random.beta(pSegs[i,j,0]*concentration, (1-pSegs[i,j,0])*concentration)
        segs = pSegs2Segs(smp, acoustic=acoustic)
    else:
        smp = np.random.uniform(size=pSegs.shape)
        segs = smp < pSegs
    return segs

def sampleSegs(pSegs, acoustic=False, resamplePSegs=False):
    segs = dict.fromkeys(pSegs)
    for doc in pSegs:
        segs[doc] = sampleSeg(pSegs[doc], acoustic, resamplePSegs)
    return segs

def pSegs2Segs(pSegs, acoustic=False, threshold=0.1, implementation='delta'):
    if acoustic:
        if implementation == 'delta':
            return relMaxWithDelta(pSegs, threshold)
        else:
            padding = [(0, 0) for d in range(len(pSegs.shape))]
            padding[1] = (1, 1)
            pSegs_padded = np.pad(pSegs, padding, 'constant', constant_values=0.)
            pSegs_padded[pSegs_padded < threshold] = 0
            segs = np.zeros_like(pSegs_padded)
            segs[argrelmax(pSegs_padded, 1)] = 1
            return segs[:, 1:-1, :]
    else:
        return pSegs > 0.5

def relMaxWithDelta(pSegs, threshold=0.1):
    segs = np.zeros_like(pSegs)
    for i in range(len(pSegs)):
        min = max = 0
        delta_left = delta_right = False
        argMax = 0
        contour = np.pad(np.squeeze(pSegs[i], -1), (0,1), 'constant', constant_values=0)
        # print(contour)
        for t in range(len(contour)):
            cur = contour[t]
            if cur > max:
                max = cur
                argMax = t
            if not delta_left:
                delta_left = max - min > threshold
            if delta_left and not delta_right:
                delta_right = max-cur > threshold
            if delta_right:
                segs[i, argMax, 0] = 1
            if delta_right or cur < min:
                min = max = cur
                delta_left = delta_right = False
                argMax = t
            # print(delta_left)
            # print(delta_right)
            # print(min)
            # print(max)
            # print('')
    return segs

def getMask(segmented):
    return np.squeeze(np.logical_not(segmented.any(-1, keepdims=True)), -1)

def getMasks(segmented):
    masks = dict.fromkeys(segmented.keys())
    for doc in masks:
        masks[doc] = getMask(segmented[doc])
    return masks

def oneHot(x, n, padCharId = 0):
    if x.shape[-1] == 1:
        x = np.squeeze(x, -1)
    out = np.equal.outer(x, np.arange(n))
    out[...,padCharId] = 0
    return out

def XsSeg2XaePhon(Xs, Xs_mask, segs, maxLen, nResample=None):
    Xae = np.split(Xs, len(Xs))
    FRAME_SIZE = Xs.shape[-1]
    deletedChars = []
    oneLetter = []
    Xae_phon = []
    for i, utt in enumerate(Xae):
        utt = np.squeeze(utt, 0)[np.logical_not(Xs_mask[i])]
        utt = np.split(utt, np.where(segs[i, :len(utt)])[0])
        if len((utt[0])) == 0:
            utt.pop(0)
        for j in range(len(utt)):
            w_len = min(len(utt[j]), maxLen)
            w_target = np.zeros((nResample if nResample else maxLen, FRAME_SIZE))
            deletedChars.append(max(0, len(utt[j]) - maxLen))
            oneLetter.append(int(w_len == 1))
            if nResample:
                if w_len > 1:
                    word = resample(utt[j][:w_len], nResample)
                else:
                    word = np.repeat(utt[j][:w_len], nResample, axis=0)
                w_len = maxLen
            else:
                word = utt[j][:w_len]
            w_target[-w_len:] = word
            Xae_phon.append(w_target)
    Xae_phon = np.stack(Xae_phon)
    deletedChars = np.array(deletedChars)
    oneLetter = np.array(oneLetter)
    return Xae_phon, deletedChars, oneLetter

def make_XAE_generator():
    pass

def XsSeg2Xae(Xs, Xs_mask, segs, maxUtt, maxLen, nResample=None, check_output=False):
    Xae = np.split(Xs, len(Xs))
    FRAME_SIZE = Xs.shape[-1]
    deletedChars = np.zeros((len(Xae), maxUtt))
    oneLetter = np.zeros((len(Xae), maxUtt))
    for i,utt in enumerate(Xae):
        utt_target = np.zeros((maxUtt, nResample if nResample else maxLen, FRAME_SIZE))
        utt = np.squeeze(utt, 0)[np.logical_not(Xs_mask[i])]
        utt = np.split(utt, np.where(segs[i,:len(utt)])[0])
        if len((utt[0])) == 0:
            utt.pop(0)
        n_words = min(len(utt), maxUtt)
        padwords = maxUtt - n_words
        for j in range(n_words):
            w_len = min(len(utt[j]), maxLen)
            w_target = np.zeros((nResample if nResample else maxLen, FRAME_SIZE))
            deletedChars[i,padwords+j] += max(0, len(utt[j]) - maxLen)
            oneLetter[i,padwords+j] += int(w_len == 1)
            if nResample:
                if w_len > 1:
                    word = resample(utt[j][:w_len], nResample)
                else:
                    word = np.repeat(utt[j][:w_len], nResample, axis=0)
                w_len = maxLen
            else:
                word = utt[j][:w_len]
            w_target[-w_len:] = word
            utt[j] = w_target
            utt_target[padwords+j] = utt[j]
        extraWDel = 0
        for j in range(maxUtt, len(utt)):
            extraWDel += len(utt[j])
        ## Uniformly distribute clipping penaresh2lty for excess words
        deletedChars[i,:] += float(extraWDel) / maxUtt
        Xae[i] = utt_target
    Xae = np.stack(Xae)
    ## NOTE: Reconstitution will fail if there has been any clipping.
    ## Do not use this feature unless maxutt and maxlen are large enough
    ## to make clipping very unlikely.
    ## Currently only works in acoustic mode.
    if check_output:
        for i in range(len(Xs)):
            src = Xs[i][np.logical_not(Xs_mask[i])]
            target = Xae[i]
            reconstituted = np.zeros((0,FRAME_SIZE))
            for wi in range(maxUtt):
                w = target[wi][np.where(target[wi].any(-1))]
                reconstituted = np.concatenate([reconstituted, w])
            for j in range(len(src)):
                assert np.allclose(src[j], reconstituted[j]), \
                       '''Reconstitution of MFCC frames failed at timestep %d.
                       Source region: %s\n Reconstituted region: %s''' \
                       %(j, src[j-1:j+2], reconstituted[j-1:j+2])

    return Xae, deletedChars, oneLetter

def XsSegs2Xae(Xs, Xs_mask, segs, maxUtt, maxLen, nResamp=None):
    Xae = []
    deletedChars = []
    oneLetter = []
    for doc in Xs:
        Xae_doc, deletedChars_doc, oneLetter_doc = XsSeg2Xae(Xs[doc],
                                                             Xs_mask[doc],
                                                             segs[doc],
                                                             maxUtt,
                                                             maxLen,
                                                             nResamp)
        Xae.append(Xae_doc)
        deletedChars.append(deletedChars_doc)
        oneLetter.append(oneLetter_doc)
    return np.concatenate(Xae), np.concatenate(deletedChars), np.concatenate(oneLetter)

def getYae(Xae, reverseUtt):
    assert len(Xae.shape) in [3,4], 'Invalid number of dimensions for Xae: %i (must be 3 or 4)' % len(Xae.shape)
    if reverseUtt:
        Yae = np.flip(Xae, 1)
        if len(Xae.shape) == 4:
            Yae = np.flip(Yae, 2)
    else:
        Yae = Xae
    return Yae

def printSegAnalysis(ae, Xs, Xs_mask, segs, maxUtt, maxLen, reverseUtt=False, batch_size=128, acoustic=False):
    Xae, deletedChars, oneLetter = XsSeg2Xae(Xs,
                                             Xs_mask,
                                             segs,
                                             maxUtt,
                                             maxLen,
                                             acoustic)

    Yae = getYae(Xae, reverseUtt)

    eval = ae.evaluate(Xae, Yae, batch_size=batch_size)
    if type(eval) is not list:
        eval = [eval]

    found_loss = eval[0]
    if not acoustic:
        found_acc = eval[1]
    found_nChar = Xae.any(-1).sum()

    print('Mean loss: %.4f' % found_loss)
    if not acoustic:
        print('Mean accuracy: %.4f' % found_acc)
    print('Deleted characters in segmentation: %d' % deletedChars.sum())
    print('Input characterrs in segmentation: %d' % found_nChar.sum())
    print()





##################################################################################
##################################################################################
##
##  METHODS FOR PROCESSING CHARACTER DATA
##
##################################################################################
##################################################################################

def text2Xs(text, maxChar, ctable):
    nUtts = len(text)
    Xs = np.zeros((nUtts, maxChar, 1))
    for ui, utt in enumerate(text):
        #Xs[ui] = ctable.encode(pad(utt[:maxChar], maxChar, "X"), maxChar)
        Xs[ui] = ctable.encode(utt[:maxChar], maxChar)
    return Xs

def texts2Xs(text, maxChar, ctable):
    utts = []
    doc_indices = dict.fromkeys(text.keys())
    ix = 0
    for doc in text:
        utts_doc = text2Xs(text[doc], maxChar, ctable)
        utts.append(utts_doc)
        doc_indices[doc] = (ix, ix + len(utts_doc))
        ix += len(utts_doc)
    return np.concatenate(utts), doc_indices

def text2Segs(text, maxChar):
    segs = []
    for utt in text:
        seg = np.zeros((maxChar,1))
        cur_ix = 0
        for w in utt:
            seg[cur_ix] = 1
            cur_ix += len(w)
            if cur_ix >= maxChar:
                break
        segs.append(seg)
    return np.stack(segs)

def texts2Segs(text, maxChar):
    segs = []
    for doc in text:
        segs_doc = text2Segs(text[doc], maxChar)
        segs.append(segs_doc)
    return np.concatenate(segs)

def batchIndices(Xt, batchSize):
    xN = Xt.shape[0]
    if xN % batchSize > 0:
        lastBit = [(xN - xN % batchSize, xN)]
    else:
        lastBit = []

    return [slice(aa, bb) for (aa, bb) in
            zip(range(0, xN, batchSize),
                range(batchSize, xN, batchSize)) + lastBit]

def charSeq2WrdSeq(segsXUtt, text):
    if type(text[0][0]) == str:
        text = ["".join(utt) for utt in text]
    else:
        text = [sum(utt, []) for utt in text]

    res = []
    for utt in range(len(text)):
        thisSeg = segsXUtt[utt]
        #pass dummy max utt length to reconstruct everything
        rText = reconstructUtt(text[utt], thisSeg, 100, wholeSent=True)
        res.append(rText)
    return res

def reconstructUtt(charSeq, segs, maxUtt, wholeSent=False):
    #print(segs)
    uttWds = np.where(segs)[0][:maxUtt]
    words = []
    s = 0
    for i in range(len(uttWds)):
        s = uttWds[i]
        if i == len(uttWds)-1:
            e = len(charSeq)
        else:
            e = uttWds[i+1]
        word = charSeq[s:e]
        words.append(word)
        # print(uttWds)
        # print(words)
        # print('')
        assert(word != "")
        s = e
    if wholeSent:
        if s < len(charSeq):
            word = charSeq[s:len(charSeq)]
            words.append(word)
    return words

def realize(rText, maxlen, maxutt):
    items = ([pad(word, maxlen, "X") for word in rText] +
             ["X" * maxlen for ii in range(maxutt - len(rText))])
    def delist(wd):
        if type(wd) == list:
            return "".join(wd)
        else:
            return wd
    items = [delist(wd) for wd in items]

    return " ".join(items)

def reconstructXs(Xs, ctable):
    reconstruction = []
    Xs_charids = Xs.argmax(-1)
    for i in range(len(Xs_charids)):
        utt = ''
        for j in range(len(Xs_charids[i])):
            utt += ctable.indices_char[int(Xs_charids[i,j])]
        reconstruction.append(utt)
    return reconstruction

def reconstructXae(Xae, ctable, maxLen=inf):
    assert len(Xae.shape) in [3,4], 'Input must be rank 3 or 4 (got rank %d)' %len(Xae.shape)
    reconstruction = []
    if len(Xae.shape) == 4:
        for i in range(len(Xae)):
            reconstruction.append([])
            for j in range(len(Xae[i])):
                word = ''
                Xae_charids = Xae[i,j][np.where(Xae[i,j].any(-1))]
                if len(Xae_charids) > 0:
                    for k in range(min(len(Xae_charids),maxLen)):
                        word += ctable.indices_char[int(Xae_charids[k])]
                    reconstruction[-1].append(word)
            reconstruction[-1] = ' '.join(reconstruction[-1])
    else:
        for j in range(len(Xae)):
            word = ''
            Xae_charids = Xae[j][np.where(Xae[j].any(-1))]
            if len(Xae_charids) > 0:
                for k in range(min(len(Xae_charids), maxLen)):
                    word += ctable.indices_char[int(Xae_charids[k])]
                reconstruction.append(word)
    return reconstruction

def printReconstruction(utt_ids, ae, Xae, ctable, batch_size=128, reverseUtt=False, maxLen=inf):
    assert len(Xae.shape) in [3,4], 'Input must be rank 3 or 4 (got rank %d)' %len(Xae.shape)
    Yae = getYae(Xae, reverseUtt)
    if len(Xae.shape) == 3:
        preds = ae.phon.predict(Xae[utt_ids], batch_size=batch_size)
    else:
        preds = ae.predict(Xae[utt_ids], batch_size=batch_size)
    input_reconstruction = reconstructXae(Xae[utt_ids], ctable, maxLen=maxLen)
    target_reconstruction = reconstructXae(Yae[utt_ids], ctable, maxLen=maxLen)
    output_reconstruction = reconstructXae(np.expand_dims(preds[range(len(utt_ids))].argmax(-1), -1), ctable, maxLen=maxLen)
    for utt in range(len(utt_ids)):
        print('Input:          %s' %input_reconstruction[utt])
        print('Target:         %s' %target_reconstruction[utt])
        print('Network:        %s' %output_reconstruction[utt])
        print('')





##################################################################################
##################################################################################
##
##  METHODS FOR PROCESSING ACOUSTIC DATA
##
##################################################################################
##################################################################################

def seg2Intervals(seg, vadIntervals):
    offset = last_interval = last_seg = 0
    this_frame = 0
    next_frame = 1
    seg_t = []
    for i in vadIntervals:
        # Interval boundaries in seconds (time)
        st, et = i
        # Interval boundaries in frames
        sf, ef = int(np.rint(float(st)*100)), int(np.rint(float(et)*100))

        offset += sf - last_interval
        last_interval = ef
        while this_frame + offset < ef:
            if next_frame >= seg.shape[0] or np.allclose(seg[next_frame], 1):
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

def segs2Intervals(seg, intervals):
    out = dict.fromkeys(intervals)
    for doc in intervals:
        out[doc] = seg2Intervals(seg[doc], intervals[doc])
    return out

def intervalsXDoc2SegsXDoc(intervals):
    out = dict.fromkeys(intervals)
    for doc in intervals:
        out[doc] = []
        for i in intervals[doc]:
            s, e = int(np.rint(float(i[0]*100))), int(np.rint(float(i[1]*100)))
            interval = [0] * (e-s)
            interval[0] = 1
            out[doc] += interval
        out[doc] = np.array(out[doc])
    return out

def getNextFrameUtt(vadSegs, maxChar, start_ix=0):
    end_ix = start_ix + 1 \
             + np.argmax(vadSegs[start_ix+1:start_ix+maxChar]) if (vadSegs[start_ix+1:start_ix+maxChar].sum() > 0) else min(len(vadSegs), start_ix+maxChar)
    return end_ix

def frameSeq2FrameSeqXUtt(raw, maxChar, vadSegs=None):
    if vadSegs is None:
        Xs = [raw[i:i+maxChar] for i in range(0, len(raw), maxChar)]
        Xs[-1] = np.pad(Xs[-1], [(0, max(0, maxChar-len(Xs[-1])))] + [(0,0) for d in Xs[-1].shape[1:]], 'constant', constant_values=0)
        out = np.stack(Xs)
    else:
        maxLen = min(len(raw), len(vadSegs))
        if len(raw) != len(vadSegs):
            print('Warning: Different number of timesteps in raw (%d) and vadBreaks (%d). Using %d.' % (len(raw), len(vadSegs), maxLen))

        oneDimInput = False
        if len(raw.shape) == 1:
            oneDimInput = True
            raw = np.expand_dims(raw, -1)
        charDim = raw.shape[-1]
        s = 0
        Xs = []
        while s < maxLen:
            e = getNextFrameUtt(vadSegs[:maxLen], maxChar, start_ix=s)
            Xs_utt = np.zeros((maxChar, charDim))
            Xs_utt[:e - s, :] = raw[s:e, :]
            Xs.append(Xs_utt)
            s = e
        out = np.stack(Xs)
        if oneDimInput:
            out = np.squeeze(Xs, -1)
    return out

def frameSeqs2FrameSeqsXUtt(raw, maxChar, doc_indices, vadSegs=None):
    Ysegs = dict.fromkeys(raw.keys())
    for doc in Ysegs:
        Ysegs[doc] = frameSeq2FrameSeqXUtt(np.expand_dims(raw[doc], -1), maxChar, vadSegs=vadSegs[doc] if vadSegs is not None else None)
    Xs = np.zeros((sum([len(Ysegs[d]) for d in Ysegs]), maxChar, 1))
    for doc in doc_indices:
        s, e = doc_indices[doc]
        Xs[s:e] = Ysegs[doc]
    return Xs

def processAcousticDocuments(raw, maxChar, vadSegs=None):
    utts = []
    doc_indices = dict.fromkeys(raw.keys())
    ix = 0
    for doc in doc_indices:
        utts_doc = frameSeq2FrameSeqXUtt(raw[doc], maxChar, vadSegs=vadSegs[doc] if vadSegs is not None else None)
        utts.append(utts_doc)
        doc_indices[doc] = (ix, ix+len(utts_doc))
        ix += len(utts_doc)
    return np.concatenate(utts), doc_indices

def filterMFCCs(mfccs, vadIntervals, charDim=40):
    # Filter out non-speech portions
    mfcc_intervals = {}
    total_frames = 0
    for doc in vadIntervals:
        mfcc_intervals[doc] = np.zeros((0, charDim))
        for i in vadIntervals[doc]:
            s, e = int(np.rint(float(i[0]*100))), int(np.rint(float(i[1]*100)))
            mfcc_intervals[doc] = np.append(mfcc_intervals[doc], mfccs[doc][s:e,:], 0)
        print('Document "%s" has %d speech frames.' %(doc, len(mfcc_intervals[doc])))
        total_frames += len(mfcc_intervals[doc])
    return mfcc_intervals, total_frames