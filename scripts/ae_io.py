from __future__ import print_function, division
import sys, re, os, math, numpy as np
from numpy import inf
from scoring import *
from data_handling import *
from echo_words import CharacterTable





##################################################################################
##################################################################################
##
##  READ METHODS
##
##################################################################################
##################################################################################

def readSegFile(timeseg_file):
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
                # In rare cases, the segmentation file
                # contains a start time earlier than previous
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

def processInputDir(dataDir, checkpoint, maxChar, ctable=None, acoustic=False, debug=False,
                    vadRegionsAsUtts=False, scoreInit=False):
    if not dataDir.endswith('/'):
        dataDir += '/'
    if acoustic:
        vadfile_paths = [dataDir + x for x in os.listdir(dataDir) if x.endswith('_vad.txt')]
        goldwrd_paths = [dataDir + x for x in os.listdir(dataDir) if x.endswith('.wrd')]
        goldphn_paths = [dataDir + x for x in os.listdir(dataDir) if x.endswith('.phn')]
        segFile = vadfile_paths[0] if len(vadfile_paths) > 0 else None
        goldWrdFile = goldwrd_paths[0] if len(goldwrd_paths) > 0 else None
        goldPhnFile = goldphn_paths[0] if len(goldphn_paths) > 0 else None
        assert segFile and (goldWrdFile or goldPhnFile), \
            'Files containing initial and gold segmentations are required in acoustic mode.'
        if 'vadIntervals' in checkpoint and 'vadSegs' in checkpoint:
            vadIntervals, vadSegs = checkpoint['vadIntervals'], checkpoint['vadSegs']
        else:
            vadIntervals, vadSegs = readSegFile(segFile)
            checkpoint['vadIntervals'] = vadIntervals
            checkpoint['vadSegs'] = vadSegs
        if 'raw' in checkpoint:
            raw = checkpoint['raw']
            total = checkpoint['total']
            FRAME_SIZE = raw[raw.keys()[0]].shape[-1]
        else:
            raw, FRAME_SIZE = readMFCCs(dataDir)
            mfccs = raw
            raw, total = filterMFCCs(raw, vadIntervals, FRAME_SIZE)
            checkpoint['raw'] = raw
            checkpoint['total'] = total
        print('Total speech frames: %s' %total)
        # for doc in raw:
        #     print(doc)
        #     print(len(mfccs[doc]))
        #     print(vadIntervals[doc][-1])
        if 'gold' in checkpoint:
            gold = checkpoint['gold']
        else:
            gold = {'wrd': None, 'phn': None}
            if goldWrdFile:
                gold['wrd'] = readGoldFrameSeg(goldWrdFile)
            if goldPhnFile:
                gold['phn'] = readGoldFrameSeg(goldPhnFile)
            checkpoint['gold'] = gold
        if goldWrdFile:
            goldWrdSegCts = 0
            for doc in gold['wrd']:
                goldWrdSegCts += len(gold['wrd'][doc])
            print('Total words: %d' % goldWrdSegCts)
            print('Mean word length: %.2f' %(float(total)/goldWrdSegCts))
        if goldPhnFile:
            goldPhnSegCts = 0
            for doc in gold['phn']:
                goldPhnSegCts += len(gold['phn'][doc])
            print('Total phonemes: %d' % goldPhnSegCts)
            print('Mean phoneme length: %.2f' %(float(total)/goldPhnSegCts))

        if scoreInit:
            if gold['wrd']:
                print('Initial word segmentation scores:')
                printSegScores(getSegScores(gold['wrd'], segs2Intervals(vadSegs, vadIntervals), 0.03, acoustic), True)
                print()
            if gold['phn']:
                print('Initial phone segmentation scores:')
                printSegScores(getSegScores(gold['phn'], segs2Intervals(vadSegs, vadIntervals), 0.02, acoustic), True)
                print()


    else:
        if 'gold' in checkpoint and 'raw' in checkpoint and 'charset' in checkpoint:
            gold, raw, charset = checkpoint['gold'], checkpoint['raw'], checkpoint['charset']
        else:
            gold, raw, charset = readTexts(dataDir, cutoff=inf)
        nChar = sum([len(u) for d in raw for u in raw[d]])
        nWrd = sum([len(w) for d in gold for w in gold[d]])
        meanLen = float(nChar)/nWrd
        print('Corpus length (characters):', nChar)
        print('Corpus length (words):', nWrd)
        print('Mean word length:', meanLen)

        if not ctable:
            ctable = CharacterTable(charset)

    doc_list = sorted(list(raw.keys()))
    segsProposal = checkpoint.get('segsProposal', [])
    checkpoint['segsProposal'] = segsProposal

    charDim = FRAME_SIZE if acoustic else ctable.dim()
    doc_list = sorted(list(raw.keys()))
    raw_cts = {}
    for doc in raw:
        if acoustic:
            raw_cts[doc] = raw[doc].shape[0]
        else:
            raw_cts[doc] = sum([len(utt) for utt in raw[doc]])
    raw_total = sum([raw_cts[doc] for doc in raw_cts])
    ## Xs: segmenter input (unsegmented input sequences by utterance)
    Xs, doc_indices = processAcousticDocuments(raw, maxChar, vadSegs=vadSegs if vadRegionsAsUtts else None) \
                      if acoustic else texts2Xs(raw, maxChar, ctable)
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
        vadSegsXUtt = frameSeqs2FrameSeqsXUtt(vadSegs, maxChar, doc_indices, vadSegs=vadSegs if vadRegionsAsUtts else None)

    if scoreInit:
        segs4evalXDoc = dict.fromkeys(doc_list)

        if acoustic:
            if goldWrdFile:
                prob = float(goldWrdSegCts)/total
                pSegs = prob * np.ones((len(Xs), maxChar, 1))
                pSegs[np.where(vadSegsXUtt)] = 1.
                pSegs[np.where(Xs_mask)] = 0.
                segs = sampleSeg(pSegs, acoustic)
                for doc in segs4evalXDoc:
                    s, e = doc_indices[doc]
                    segs4evalXDoc[doc] = segs[s:e]
                    if acoustic:
                        masked_proposal = np.ma.array(segs4evalXDoc[doc], mask=Xs_mask[s:e])
                        segs4evalXDoc[doc] = masked_proposal.compressed()
                segs4evalXDoc = segs2Intervals(segs4evalXDoc, vadIntervals)
                print('Random segmentations at same rate as gold words:')
                printSegScores(getSegScores(gold['wrd'], segs4evalXDoc, tol=.03, acoustic=acoustic), acoustic=acoustic)
                _, goldseg = readSegFile(goldWrdFile)
                Y = frameSeqs2FrameSeqsXUtt(goldseg, maxChar, doc_indices, vadSegs=vadSegs if vadRegionsAsUtts else None)
                for doc in segs4evalXDoc:
                    s, e = doc_indices[doc]
                    segs4evalXDoc[doc] = Y[s:e]
                    if acoustic:
                        masked_proposal = np.ma.array(segs4evalXDoc[doc], mask=Xs_mask[s:e])
                        segs4evalXDoc[doc] = masked_proposal.compressed()
                segs4evalXDoc = segs2Intervals(segs4evalXDoc, vadIntervals)
                print('Gold word segmentations:')
                printSegScores(getSegScores(gold['wrd'], segs4evalXDoc, tol=.03, acoustic=acoustic), acoustic=acoustic)
            if goldPhnFile:
                prob = float(goldPhnSegCts)/total
                pSegs = prob * np.ones((len(Xs), maxChar, 1))
                pSegs[np.where(vadSegsXUtt)] = 1.
                pSegs[np.where(Xs_mask)] = 0.
                segs = sampleSeg(pSegs, acoustic)
                for doc in segs4evalXDoc:
                    s, e = doc_indices[doc]
                    segs4evalXDoc[doc] = segs[s:e]
                    if acoustic:
                        masked_proposal = np.ma.array(segs4evalXDoc[doc], mask=Xs_mask[s:e])
                        segs4evalXDoc[doc] = masked_proposal.compressed()
                segs4evalXDoc = segs2Intervals(vadIntervals, segs4evalXDoc)
                print('Random segmentations at same rate as gold phones:')
                printSegScores(getSegScores(gold['phn'], segs4evalXDoc, tol=.02, acoustic=acoustic), acoustic=acoustic)
                _, goldseg = readSegFile(goldPhnFile)
                Y = frameSeqs2FrameSeqsXUtt(goldseg, maxChar, doc_indices, vadSegs=vadSegs if vadRegionsAsUtts else None)
                for doc in segs4evalXDoc:
                    s, e = doc_indices[doc]
                    segs4evalXDoc[doc] = Y[s:e]
                    if acoustic:
                        masked_proposal = np.ma.array(segs4evalXDoc[doc], mask=Xs_mask[s:e])
                        segs4evalXDoc[doc] = masked_proposal.compressed()
                segs4evalXDoc = segs2Intervals(segs4evalXDoc, vadIntervals)
                print('Gold phone segmentation')
                printSegScores(getSegScores(gold['phn'], segs4evalXDoc, tol=.03, acoustic=acoustic), acoustic=acoustic)
        else:
            prob = float(1) / meanLen
            pSegs = prob * np.ones((len(Xs), maxChar, 1))
            pSegs[:, 0] = 1.
            pSegs[np.where(Xs_mask)] = 0.
            segs = sampleSeg(pSegs, acoustic)
            for doc in segs4evalXDoc:
                s, e = doc_indices[doc]
                segs4evalXDoc[doc] = segs[s:e]
                if acoustic:
                    masked_proposal = np.ma.array(segs4evalXDoc[doc], mask=Xs_mask[s:e])
                    segs4evalXDoc[doc] = masked_proposal.compressed()
            printSegScores(getSegScores(gold, segs4evalXDoc, acoustic=acoustic), acoustic=acoustic)

    return doc_indices, doc_list, charDim, raw_total, Xs, Xs_mask, gold, (vadIntervals, vadSegs, vadSegsXUtt, segFile, goldWrdFile, goldPhnFile) if acoustic else ctable

def readText(path, cutoff=inf):
    i = 0
    lines = []
    with open(path, 'rb') as f:
        line = f.readline()
        while line and i < cutoff:
            lines.append(line)
            line = f.readline()
            i += 1
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

def readTexts(path, cutoff=inf):
    if not path.endswith('/'):
        path += '/'
    basename = re.compile('.*/(.+)\.txt')
    filelist = sorted([path+x for x in os.listdir(path) if x.endswith('.txt')])
    idlist = [basename.match(x).group(1) for x in filelist]

    text = {}
    chars = {}
    charsets = {}

    for i in xrange(len(filelist)):
        text[idlist[i]], chars[idlist[i]], charsets[idlist[i]] = readText(filelist[i], cutoff)
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





##################################################################################
##################################################################################
##
##  WRITE METHODS
##
##################################################################################
##################################################################################

def writeLog(batch_num_global, iteration, epochAELoss, epochAcc, epochSegLoss, epochDel, epochOneL, epochSeg, gold,
             segsProposal, logdir, vadIntervals=None, acoustic=False, print_headers=False, filename='log.txt'):
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
        segsProposal = segs2Intervals(segsProposal, vadIntervals)
        scores = {'wrd': None, 'phn': None}
        if gold['wrd']:
            headers += ['bp_wrd', 'br_wrd', 'bf_wrd', 'swp_wrd', 'swr_wrd', 'swf_wrd']
            scores['wrd'] = getSegScores(gold['wrd'], segsProposal, tol = .03, acoustic=True)
        if gold['phn']:
            headers += ['bp_phn', 'br_phn', 'bf_phn', 'swp_phn', 'swr_phn', 'swf_phn']
            scores['phn'] = getSegScores(gold['phn'], segsProposal, tol = .02, acoustic=True)
        for doc in set(scores['wrd'].keys() + scores['phn'].keys()):
            if not doc == '##overall##':
                score_row = outVar[:]
                if gold['wrd'] and scores['wrd'][doc]:
                    _, (bp,br,bf), _, (swp,swr,swf) = scores['wrd'][doc]
                    score_row += [bp, br, bf, swp, swr, swf]
                if gold['phn'] and scores['phn'][doc]:
                    _, (bp,br,bf), _, (swp,swr,swf) = scores['phn'][doc]
                    score_row += [bp, br, bf, swp, swr, swf]
                with open(logdir+'/'+doc+'_'+filename, 'ab') as f:
                    if print_headers:
                        print("\t".join(headers), file=f)
                    print("\t".join(["%g" % xx for xx in score_row]), file=f)
        with open(logdir+'/'+filename, 'ab') as f:
            if print_headers:
                print("\t".join(headers), file=f)
            score_row = outVar[:]
            if gold['wrd']:
                _, (bp,br,bf), _, (swp,swr,swf) = scores['wrd']['##overall##']
                score_row += [bp, br, bf, swp, swr, swf]
            if gold['phn']:
                _, (bp,br,bf), _, (swp,swr,swf) = scores['phn']['##overall##']
                score_row += [bp, br, bf, swp, swr, swf]
            print("\t".join(["%g" % xx for xx in score_row]), file=f)
        return scores
    else:
        headers += ["bp", "br", "bf", "swp", "swr", "swf", "lp", "lr", "lf"]
        score_row = outVar[:]
        for doc in segsProposal:
            segmented = charSeq2WrdSeq(segsProposal[doc], gold[doc])
            (bp,br,bf) = scoreBreaks(gold[doc], segmented)
            score_row += [bp,br,bf]
            (swp,swr,swf) = scoreWords(gold[doc], segmented)
            score_row += [swp,swr,swf]
            (lp,lr,lf) = scoreLexicon(gold[doc], segmented)
            score_row += [lp,lr,lf]
            with open(logdir+'/'+filename, 'ab') as f:
                if print_headers:
                    print("\t".join(headers), file=f)
                print("\t".join(["%g" % xx for xx in score_row]), file=f)

## Used only in text mode
def writeCharSegs(logdir, allBestSeg, text, iteration=None, filename='seg.txt'):
    segmented = charSeq2WrdSeq(allBestSeg, text)

    if iteration:
        logfile = file(logdir + '/' + str(iteration) + '_' + filename, 'w')
    else:
        logfile = file(logdir + '/' + filename, 'w')
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

def writeTimeSegs(segs, out_dir='./', TextGrid=False, dataset='train'):
    assert type(out_dir) == str, 'out_file must be a directory path.'
    out_dir = out_dir + '/'
    if TextGrid:
        suffix = '_' + dataset + '.TextGrid'
    else:
        suffix = 'seg_' + dataset + '.txt'
    for doc in segs:
        with open(out_dir + doc + suffix, 'wb') as f:
            writeTimeSeg(segs[doc], out_file=f, docname=doc, TextGrid=TextGrid)