from __future__ import print_function, division
import sys, re, numpy as np
from echo_words import CharacterTable, pad

## GENERAL METHODS
def segs2pSegs(segs, alpha=0.05):
    assert alpha >= 0.0 and alpha <= 0.5, 'Illegal value of alpha (0 <= alpha <= 0.5)'
    return np.maximum(alpha, segs - alpha)

def getMask(segmented):
    return np.squeeze(np.logical_not(segmented.any(-1, keepdims=True)), -1)

def getMasks(segmented):
    masks = dict.fromkeys(segmented.keys())
    for doc in masks:
        masks[doc] = getMask(segmented[doc])
    return masks

def XsSeg2Xae(Xs, Xs_mask, segs, maxUtt, maxLen, check_output=False):
    Xae = np.split(Xs, len(Xs))
    FRAME_SIZE = Xs.shape[-1]
    deletedChars = np.zeros((len(Xae), maxUtt))
    oneLetter = np.zeros(len(Xae))
    for i,utt in enumerate(Xae):
        utt_target = np.zeros((maxUtt, maxLen, FRAME_SIZE))
        utt = np.squeeze(utt, 0)[np.logical_not(Xs_mask[i])]
        utt = np.split(utt, np.where(segs[i,:len(utt)])[0])
        if len((utt[0])) == 0:
            utt.pop(0)
        n_words = min(len(utt), maxUtt)
        for j in xrange(n_words):
            w_len = min(len(utt[j]), maxLen)
            w_target = np.zeros((maxLen, FRAME_SIZE))
            deletedChars[i,j] += max(0, len(utt[j]) - maxLen)
            oneLetter[i] += int(w_len == 1)
            w_target[:w_len,:] = utt[j][:w_len]
            utt[j] = w_target
            utt_target[j,:] = utt[j]
        extraWDel = 0
        for j in xrange(maxUtt, len(utt)):
            extraWDel += len(utt[j])
        ## Uniformly distribute clipping penalty for excess words
        deletedChars[i,:] += float(extraWDel) / maxUtt
        Xae[i] = utt_target
    Xae = np.stack(Xae)
    ## NOTE: Reconstitution will fail if there has been any clipping.
    ## Do not use this feature unless maxutt and maxlen are large enough
    ## to make clipping very unlikely.
    if check_output:
        for i in xrange(len(Xs)):
            src = Xs[i][np.logical_not(Xs_mask[i])]
            target = Xae[i]
            reconstituted = np.zeros((0,FRAME_SIZE))
            for wi in xrange(maxUtt):
                w = target[wi][np.where(target[wi].any(-1))]
                reconstituted = np.concatenate([reconstituted, w])
            for j in xrange(len(src)):
                assert np.allclose(src[j], reconstituted[j]), \
                       '''Reconstitution of MFCC frames failed at timestep %d.
                       Source region: %s\n Reconstituted region: %s''' \
                       %(j, src[j-1:j+2], reconstituted[j-1:j+2])

    return Xae, deletedChars, oneLetter

def XsSegs2Xae(Xs, Xs_mask, segs, maxUtt, maxLen):
    X = dict.fromkeys(Xs.keys())
    deletedChars = dict.fromkeys(Xs.keys())
    oneLetter = dict.fromkeys(Xs.keys())
    for doc in Xs:
        X[doc], deletedChars[doc], oneLetter[doc] = XsSeg2Xae(Xs[doc],
                                                              Xs_mask[doc],
                                                              segs[doc],
                                                              maxUtt,
                                                              maxLen)
    return X, deletedChars, oneLetter





## TEXT DATA
def text2Xs(text, maxChar, ctable):
    nUtts = len(text)
    Xs = np.zeros((nUtts, maxChar, ctable.dim()), dtype=np.bool)
    for ui, utt in enumerate(text):
        Xs[ui] = ctable.encode(utt[:maxChar], maxChar)
    return Xs

def texts2Xs(text, maxChar, ctable):
    Xs = dict.fromkeys(text.keys())
    for doc in text:
        Xs[doc] = text2Xs(text[doc], maxChar, ctable)
    return Xs

def text2Segs(text, maxChar):
    nUtts = len(text)
    segs = []
    for utt in text:
        seg = np.zeros((maxChar,1))
        cur_ix = -1
        for w in utt:
            cur_ix += len(w)
            if cur_ix >= maxChar:
                break
            seg[cur_ix] = 1
        segs.append(seg)
    return np.stack(segs)

def texts2Segs(text, maxChar):
    segs = dict.fromkeys(text.keys())
    for doc in text:
        segs[doc] = text2Segs(text[doc], maxChar)
    return segs

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





## ACOUSTIC DATA
def timeSegs2frameSegs(timeseg_file):
    intervals = {}
    speech = {}
    offsets = {}
    seg = 0
    with open(timeseg_file, 'rb') as f:
        lines = f.readlines()
        lines.sort(key = lambda x: float(x.strip().split()[1]))
    for line in lines:
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

def frameInput2Utts(raw, vadBreaks, maxChar):
    frame_len = raw.shape[-1]
    s = 0
    Xs = []
    while s < raw.shape[0]:
        s, e = getNextFrameUtt(vadBreaks, maxChar, start_ix=s)
        Xseg_batch = np.zeros((maxChar, frame_len))
        Xseg_batch[:e - s, :] = raw[s:e, :]
        Xs.append(Xseg_batch)
        s = e
    return np.stack(Xs)

def frameInputs2Utts(raw, vadBreaks, maxChar):
    Xs = dict.fromkeys(raw.keys())
    for doc in Xs:
        Xs[doc] = frameInput2Utts(raw[doc], vadBreaks[doc], maxChar)
    return Xs

def frameSeg2FrameSegXUtt(framesegs, vadBreaks, maxChar):
    s = 0
    Yseg = []
    while s < framesegs.shape[0]:
        s, e = getNextFrameUtt(vadBreaks, maxChar, start_ix=s)
        Yseg_batch = np.zeros(maxChar)
        Yseg_batch[:e - s] = framesegs[s:e]
        Yseg.append(Yseg_batch)
        s = e
    Yseg = np.stack(Yseg)
    return Yseg

def frameSegs2FrameSegsXUtt(framesegs, vadBreaks, maxChar):
    doc_list = list(framesegs.keys())
    Yseg = dict.fromkeys(doc_list)
    for doc in doc_list:
        Yseg[doc] = frameSeg2FrameSegXUtt(framesegs[doc], vadBreaks[doc], maxChar)
    return Yseg

def intervals2ForcedSeg(intervals):
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

def getNextFrameUtt(vad_breaks, BATCH_SIZE, start_ix=0):
    if start_ix + BATCH_SIZE >= len(vad_breaks):
        return start_ix, len(vad_breaks)
    end_ix = start_ix + 1 + np.argmax(vad_breaks[start_ix+1:]) ## Gets first occurrence
    if end_ix > start_ix + BATCH_SIZE:
        return start_ix, start_ix + BATCH_SIZE
    return start_ix, end_ix

def seg2pSegWithForced(segs, vad_breaks, alpha=0.05):
    assert alpha >= 0.0 and alpha <= 0.5, 'Illegal value of alpha (0 <= alpha <= 0.5)'
    out = segs2pSegs(segs, alpha)
    out[np.where(vad_breaks)] = 1.
    return out

def segs2pSegsWithForced(segs, vad_breaks, alpha=0.05):
    out = dict.fromkeys(segs.keys())
    for doc in segs:
        out[doc] = seg2pSegWithForced(segs[doc], vad_breaks[doc], alpha=0.05)
    return out

def filterMFCCs(mfccs, intervals, segs, FRAME_SIZE=40):
    # Filter out non-speech portions
    mfcc_intervals = {}
    total_frames = 0
    for doc in segs:
        mfcc_intervals[doc] = np.zeros((0,FRAME_SIZE))
        for i in intervals[doc]:
            sf, ef = int(np.rint(float(i[0]*100))), int(np.rint(float(i[1]*100)))
            mfcc_intervals[doc] = np.append(mfcc_intervals[doc], mfccs[doc][sf:ef,:], 0)
        print('Document "%s" has %d speech frames.' %(doc, len(mfcc_intervals[doc])))
        total_frames += len(mfcc_intervals[doc])
    print('Complete dataset has %d speech frames.' %total_frames)
    return mfcc_intervals
