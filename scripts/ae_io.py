from __future__ import print_function, division
import sys, re, os, math, numpy as np
from scoring import *
from data_handling import charSeq2WrdSeq, frameSegs2timeSegs, timeSegs2frameSegs, intervals2ForcedSeg, filterMFCCs, \
frameInputs2Utts, frameSegs2FrameSegsXUtt, texts2Xs, getMask, reconstructXs
from echo_words import CharacterTable

## READ METHODS
def processInputDir(dataDir, checkpoint, maxChar, acoustic=False, debug=False):
    if acoustic:
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
            print('Gold word segmentation count: %d' % goldWrdSegCts)
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

        if gold['wrd'] and not args.supervisedSegmenter:
            print('Initial word segmentation scores:')
            printSegScores(getSegScores(gold['wrd'], frameSegs2timeSegs(intervals, segs_init), acoustic), True)
            print()
        if gold['phn'] and not args.supervisedSegmenter:
            print('Initial phone segmentation scores:')
            printSegScores(getSegScores(gold['phn'], frameSegs2timeSegs(intervals, segs_init), acoustic), True)
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
    segsProposal = checkpoint.get('segsProposal', [])
    checkpoint['segsProposal'] = segsProposal

    charDim = FRAME_SIZE if acoustic else len(charset)
    doc_list = sorted(list(raw.keys()))
    raw_cts = {}
    for doc in raw:
        if acoustic:
            raw_cts[doc] = raw[doc].shape[0]
        else:
            raw_cts[doc] = sum([len(utt) for utt in raw[doc]])
    raw_total = sum([raw_cts[doc] for doc in raw_cts])
    ## Xs: segmenter input (unsegmented input sequences by utterance)
    Xs, doc_indices = frameInputs2Utts(raw, vadBreaks, maxChar) if acoustic else texts2Xs(raw, maxChar, ctable)
    if debug and not acoustic:
        n = 20
        print('Character reconstruction check:')
        for doc in doc_list:
            s = doc_indices[doc][0]
            print('Document: %s' % doc)
            reconstruction = reconstructXs(Xs[s:n], ctable)
            for i in range(n):
                print('Input string:   %s' % raw[doc][i])
                print('Reconstruction: %s' % reconstruction[i])

    ## Xs_mask: mask of padding timesteps by utterance
    if acoustic:
        Xs_mask = getMask(Xs)
    else:
        Xs_mask = np.zeros((len(Xs), maxChar))
        for doc in doc_list:
            s, e = doc_indices[doc]
            Xs_mask_doc = np.zeros((e - s, maxChar))
            for i in range(len(raw[doc])):
                utt_len = len(raw[doc][i])
                Xs_mask_doc[i][utt_len:] = 1
            Xs_mask[s:e] = Xs_mask_doc

    if acoustic:
        vad = frameSegs2FrameSegsXUtt(vadBreaks, vadBreaks, maxChar, doc_indices)

    return doc_indices, doc_list, charDim, raw_total, Xs, Xs_mask, gold, (intervals, vad, vadBreaks, SEGFILE, GOLDWRD, GOLDPHN) if acoustic else ctable

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
def writeLog(batch_num_global, iteration, epochAELoss, epochAcc, epochSegLoss, epochDel, epochOneL, epochSeg, text, segsProposal, logdir, intervals=None, acoustic=False, print_headers=False):
    headers = ['Batch', 'Iteration']
    outVar = [batch_num_global, iteration]
    if epochAELoss != None:
        headers.append('AELoss')
        outVar.append(epochAELoss)
    if epochSegLoss != None:
        headers.append('SegmenterLoss')
        outVar.append(epochSegLoss)
    if epochAcc != None:
        headers.append('Accuracy')
        outVar.append(epochAcc)
    if epochDel != None:
        headers.append('NumDel')
        outVar.append(epochDel)
    if epochOneL != None:
        headers.append('NumOneL')
        outVar.append(epochOneL)
    headers.append('NumSeg')
    outVar.append(epochSeg)
    if acoustic:
        segsProposal = frameSegs2timeSegs(intervals, segsProposal)
        scores = {'wrd': None, 'phn': None}
        if text['wrd']:
            headers += ['bp_wrd', 'br_wrd', 'bf_wrd', 'swp_wrd', 'swr_wrd', 'swf_wrd']
            scores['wrd'] = getSegScores(text['wrd'], segsProposal, tol = .03, acoustic=True)
        if text['phn']:
            headers += ['bp_phn', 'br_phn', 'bf_phn', 'swp_phn', 'swr_phn', 'swf_phn']
            scores['phn'] = getSegScores(text['phn'], segsProposal, tol = .02, acoustic=True)
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