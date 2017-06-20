from __future__ import print_function, division
import sys, re, os, math, numpy as np
from scoring import *
from data_handling import charSeq2WrdSeq, frameSegs2timeSegs

## READ METHODS
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
        charset = set("".join(chars + ['X']))

    return text, chars, charset

def readTexts(path):
    if not path.endswith('/'):
        path += '/'
    basename = re.compile('.*/(.+)\.txt')
    filelist = sorted([path+x for x in os.listdir(path) if x.endswith('.txt')])
    idlist = [basename.match(x).group(1) for x in filelist]

    text = {}
    chars = {}
    charsets = {}

    for i in xrange(len(filelist)):
        text[idlist[i]], chars[idlist[i]], charsets[idlist[i]] = readText(filelist[i])
    charset = list(set.union(*[charsets[d] for d in charsets]))
    return text, chars, charset

def readGoldFrameSeg(path):
    gold_seg = {}
    with open(path, 'rb') as gold: 
        for line in gold: 
            if line.strip() != '': 
                doc, start, end = line.strip().split()[:3] 
                if doc in gold_seg: 
                    gold_seg[doc].append((float(start),float(end))) 
                else: 
                    gold_seg[doc] = [(float(start),float(end))] 
    for doc in gold_seg: 
        gold_seg[doc].sort(key=lambda x: x[0]) 
    return gold_seg 


def readMFCCs(path, filter_file=None):
    if not path.endswith('/'):
        path += '/'
    basename = re.compile('.*/(.+)\.mfcc')
    filelist = sorted([path+x for x in os.listdir(path) if x.endswith('.mfcc')])
    idlist = [basename.match(x).group(1) for x in filelist]

    if filter_file:
        to_keep = {}
        with open(filter_file, 'rb') as s:
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
                        mfcc = map(float, line.strip().split())
                    if first:
                        FRAME_SIZE = len(mfcc)
                        first = False
                    assert len(mfcc) == FRAME_SIZE, 'ERROR: MFCC size (%d) in file "%s" does not agree with inferred frame size (%d).' %(len(mfcc),p,FRAME_SIZE)
                    mfcc_lists[file_id].append(mfcc)
        mfcc_lists[file_id] = np.asarray(mfcc_lists[file_id])
    return mfcc_lists, FRAME_SIZE



## WRITE METHODS
def writeLog(iteration, epochAELoss, epochAcc, epochSegLoss, epochDel, epochOneL, epochSeg, text, segsProposal, logdir, intervals=None, acoustic=False, print_headers=False):
    headers = ['iteration']
    outVar = [iteration]
    if epochAELoss != None:
        headers.append('epochAELoss')
        outVar.append(epochAELoss)
    if epochSegLoss != None:
        headers.append('epochSegLoss')
        outVar.append(epochSegLoss)
    if epochAcc != None:
        headers.append('epochAcc')
        outVar.append(epochAcc)
    if epochDel != None:
        headers.append('epochDel')
        outVar.append(epochDel)
    if epochOneL != None:
        headers.append('epochOneL')
        outVar.append(epochOneL)
    headers.append('epochSeg')
    outVar.append(epochSeg)
    if acoustic:
        segsProposal = frameSegs2timeSegs(intervals, segsProposal)
        scores = {'wrd': None, 'phn': None}
        if text['wrd']:
            headers += ['bp_wrd', 'br_wrd', 'bf_wrd', 'swp_wrd', 'swr_wrd', 'swf_wrd']
            scores['wrd'] = getSegScores(text['wrd'], segsProposal, acoustic=True)
        if text['phn']:
            headers += ['bp_phn', 'br_phn', 'bf_phn', 'swp_phn', 'swr_phn', 'swf_phn']
            scores['phn'] = getSegScores(text['phn'], segsProposal, acoustic=True)
        for doc in set(scores['wrd'].keys() + scores['phn'].keys()):
            if not doc == '##overall##':
                score_row = outVar[:]
                if text['wrd'] and scores['wrd'][doc]:
                    _, (bp,br,bf), _, (swp,swr,swf) = scores['wrd'][doc]
                    score_row += [bp, br, bf, swp, swr, swf]
                if text['phn'] and scores['phn'][doc]:
                    _, (bp,br,bf), _, (swp,swr,swf) = scores['phn'][doc]
                    score_row += [bp, br, bf, swp, swr, swf]
                with open(logdir+'/'+doc+'_log.txt', 'ab') as f:
                    if print_headers:
                        print("\t".join(headers), file=f)
                    print("\t".join(["%g" % xx for xx in score_row]), file=f)
        with open(logdir+'/log.txt', 'ab') as f:
            if print_headers:
                print("\t".join(headers), file=f)
            score_row = outVar[:]
            if text['wrd']:
                _, (bp,br,bf), _, (swp,swr,swf) = scores['wrd']['##overall##']
                score_row += [bp, br, bf, swp, swr, swf]
            if text['phn']:
                _, (bp,br,bf), _, (swp,swr,swf) = scores['phn']['##overall##']
                score_row += [bp, br, bf, swp, swr, swf]
            print("\t".join(["%g" % xx for xx in score_row]), file=f)
        return scores
    else:
        headers += ["bp", "br", "bf", "swp", "swr", "swf", "lp", "lr", "lf"]
        score_row = outVar[:]
        for doc in segsProposal:
            segmented = charSeq2WrdSeq(segsProposal[doc], text[doc])
            (bp,br,bf) = scoreBreaks(text[doc], segmented)
            score_row += [bp,br,bf]
            (swp,swr,swf) = scoreWords(text[doc], segmented)
            score_row += [swp,swr,swf]
            (lp,lr,lf) = scoreLexicon(text[doc], segmented)
            score_row += [lp,lr,lf]
            with open(logdir+'/log.txt', 'ab') as f:
                if print_headers:
                    print("\t".join(headers), file=f)
                print("\t".join(["%g" % xx for xx in score_row]), file=f)

## Used only in text mode
def writeSolutions(logdir, allBestSeg, text, iteration):
    segmented = charSeq2WrdSeq(allBestSeg, text)

    logfile = file(logdir + "/segmented-%d.txt" % iteration, 'w')
    if type(text[0][0]) == str:
        for line in segmented:
            print(" ".join(line), file=logfile)
    else:
        for line in segmented:
            print(" || ".join([" ".join(wi) for wi in line]), "||",
                  file=logfile)
    logfile.close()

## Used only in acoustic mode
def writeTimeSeg(seg, out_file=sys.stdout, docname=None, TextGrid=False):
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

def writeTimeSegs(segs, out_dir='./', TextGrid=False):
    assert type(out_dir) == str, 'out_file must be a directory path.'
    out_dir = out_dir + '/'
    if TextGrid:
        suffix = '.TextGrid'
    else:
        suffix = '_seg.txt'
    for doc in segs:
        with open(out_dir + doc + suffix, 'wb') as f:
            writeTimeSeg(segs[doc], out_file=f, docname=doc, TextGrid=TextGrid)