from __future__ import print_function, division
from data_handling import *
from sampling import *
from ae_io import readSegFile
from network import *
import time





##################################################################################
##################################################################################
##
##  TEST AE ON FIXED SEGMENTATION
##
##################################################################################
##################################################################################

def testFullAEOnFixedSeg(ae, Xs, Xs_mask, pSegs, maxChar, maxUtt, maxLen, doc_indices, utt_ids, logdir,
                         reverseUtt=False, batch_size=128, nResample=None, trainIters=100, vadSegs=None, ctable=None,
                         gold=None, segLevel=None, acoustic=False, fitParts=True, fitFull=True, vadRegionsAsUtts=False,
                         debug=False):
    assert acoustic or ctable is not None, 'ctable object must be provided to testFullAEOnFixedSeg() in character mode'
    assert not acoustic or vadSegs is not None, 'vadBreaks object must be provided to testFullAEOnFixedSeg() in acoustic mode'

    flatDecoder = ae.flatDecoder

    if gold is not None:
        if acoustic:
            _, goldseg = readSegFile(gold)
            Y = frameSeqs2FrameSeqsXUtt(goldseg, maxChar, doc_indices, vadSegs=vadSegs if vadRegionsAsUtts else None)
        else:
            Y = texts2Segs(gold, maxChar)
    else:
        Y = sampleSeg(pSegs)

    print()
    print('Unit testing full auto-encoder on fixed %s segmentation' % ('gold' if gold != None else 'random'))
    ## Re-initialize network weights in case any training has already happened
    ae.load(logdir, suffix='_init')

    logfile = logdir + '/log_fixedfull_' + segLevel + '.txt'
    headers = ['Iteration', 'AEFullLoss']
    if not acoustic:
        headers += ['AEFullAcc']
    with open(logfile, 'wb') as f:
        print('\t'.join(headers), file=f)

    for i in range(trainIters):
        t0 = time.time()
        print('Iteration %d' % (i + 1))
        Xae, _, _, eval = ae.trainFullOnFixed(Xs,
                                              Xs_mask,
                                              Y,
                                              maxUtt,
                                              maxLen,
                                              batch_size=batch_size,
                                              reverseUtt=reverseUtt,
                                              nResample=nResample,
                                              fitParts=fitParts,
                                              fitFull=fitFull)

        if nResample:
            Xae_full, _, _ = XsSeg2Xae(Xs[utt_ids],
                                       Xs_mask[utt_ids],
                                       Y[utt_ids],
                                       maxUtt,
                                       maxLen,
                                       nResample=None)
        prefix = 'fixedfull'
        if segLevel:
            prefix += segLevel

        ae.plotFull(Xae_full if nResample else Xae[utt_ids],
                    getYae(Xs[utt_ids] if flatDecoder else Xae[utt_ids], reverseUtt),
                    logdir,
                    prefix,
                    i + 1,
                    batch_size=batch_size,
                    Xae_resamp=Xae if nResample else None,
                    debug=debug)

        if not acoustic:
            print()
            print('Example reconstructions')
            printReconstruction(utt_ids, ae, Xae, ctable, batch_size, reverseUtt)

            if ae.latentDim == 3:
                ae.plotVAE(logdir,
                           prefix,
                           ctable=ctable,
                           reverseUtt=reverseUtt,
                           batch_size=batch_size,
                           debug=debug)

        row = [str(i+1), str(eval[0])]
        if not acoustic:
            row += [str(eval[1])]
        with open(logfile, 'ab') as f:
            print('\t'.join(row), file=f)

        t1 = time.time()
        print('Iteration time: %.2fs' %(t1-t0))
        print()





##################################################################################
##################################################################################
##
##  TEST PHONOLOGICAL AE ON FIXED SEGMENTATION
##
##################################################################################
##################################################################################

def testPhonAEOnFixedSeg(ae, Xs, Xs_mask, pSegs, maxChar, maxLen, doc_indices, utt_ids, logdir, reverseUtt=False,
                         batch_size=128, nResample=None, trainIters=100, vadSegs=None, ctable=None, gold=None,
                         segLevel=None, acoustic=False, vadRegionsAsUtts=False, debug=False):

    flatDecoder = ae.flatDecoder

    if gold is not None:
        if acoustic:
            _, goldseg = readSegFile(gold)
            Y = frameSeqs2FrameSeqsXUtt(goldseg, maxChar, doc_indices, vadSegs=vadSegs if vadRegionsAsUtts else None)
        else:
            Y = texts2Segs(gold, maxChar)
    else:
        Y = sampleSeg(pSegs)

    print()
    print('Unit testing phonological auto-encoder on fixed %s segmentation' % ('gold' if gold is not None else 'random'))
    ## Re-initialize network weights in case any training has already happened
    ae.load(logdir, suffix='_init')

    logfile = logdir + '/log_fixedphon_' + segLevel + '.txt'
    headers = ['Iteration', 'AEPhonLoss']
    if not acoustic:
        headers += ['AEPhonAcc']
    with open(logfile, 'wb') as f:
        print('\t'.join(headers), file=f)

    for i in range(trainIters):
        t0 = time.time()
        print('Iteration %d' % (i + 1))

        Xae, deletedChars, oneLetter, eval = ae.trainPhonOnFixed(Xs,
                                                                 Xs_mask,
                                                                 Y,
                                                                 maxLen,
                                                                 reverseUtt=reverseUtt,
                                                                 batch_size=batch_size,
                                                                 nResample=nResample)

        Yae = getYae(Xs if ae.flatDecoder else Xae, reverseUtt)

        if nResample:
            Xae_full, _, _ = XsSeg2XaePhon(Xs[utt_ids],
                                           Xs_mask[utt_ids],
                                           Y[utt_ids],
                                           maxLen,
                                           nResample=None)

        prefix = 'fixedphon'
        if segLevel:
            prefix += segLevel

        ae.plotPhon(Xae_full if nResample else Xae[utt_ids],
                    Yae[utt_ids],
                    logdir,
                    prefix,
                    i + 1,
                    batch_size,
                    Xae_resamp=Xae[utt_ids] if nResample else None,
                    debug=debug)

        if not acoustic:
            print()
            print('Example reconstructions')
            printReconstruction(utt_ids, ae, Xae, ctable, batch_size, reverseUtt)

            if ae.latentDim == 3:
                ae.plotVAE(logdir,
                           prefix,
                           ctable=ctable,
                           reverseUtt=reverseUtt,
                           batch_size=batch_size,
                           debug=debug)

        row = [str(i+1), str(eval[0])]
        if not acoustic:
            row += [str(eval[1])]
        with open(logfile, 'ab') as f:
            print('\t'.join(row), file=f)

        t1 = time.time()

        print('Iteration time: %.2fs' %(t1-t0))
        print()





##################################################################################
##################################################################################
##
##  TEST SEGMENTER ON FIXED SEGMENTATION
##
##################################################################################
##################################################################################

def testSegmenterOnFixedSeg(segmenter, Xs, Xs_mask, pSegs, maxChar, doc_indices, utt_ids, logdir, trainIters=100,
                            batch_size=128, vadSegs=None, vadSegsXUtt=None, vadIntervals=None, gold=None, goldEval=None,
                            segLevel=None, acoustic=False, vadRegionsAsUtts=False, debug=False):
    if gold:
        if acoustic:
            _, goldseg = readSegFile(gold)
            Y = frameSeqs2FrameSeqsXUtt(goldseg, maxChar, doc_indices, vadSegs=vadSegs if vadRegionsAsUtts else None)
        else:
            Y = texts2Segs(gold, maxChar)
    else:
        Y = sampleSeg(pSegs)

    print()
    print('Unit testing segmenter network on fixed %s segmentation' % ('gold' if gold is not None else 'random'))

    proposalsXDoc = dict.fromkeys(doc_indices.keys())
    targetsXDoc = dict.fromkeys(doc_indices)

    ## Re-initialize network weights in case any training has already happened
    segmenter.load(logdir, suffix='_init')

    for i in range(trainIters):
        t0 = time.time()
        print('Iteration %d' % (i + 1))

        segmenter.trainOnFixed(Xs,
                               Xs_mask,
                               Y,
                               batch_size=batch_size)

        print('Getting model predictions for evaluation')
        preds = segmenter.predict(Xs,
                                  Xs_mask,
                                  batch_size=batch_size)

        segsProposal = (preds > 0.5) * np.expand_dims(1-Xs_mask, -1)

        if acoustic:
            segsProposal[np.where(vadSegsXUtt)] = 1.
        else:
            segsProposal[:, 0, ...] = 1.
        for doc in proposalsXDoc:
            s, e = doc_indices[doc]
            proposalsXDoc[doc] = segsProposal[s:e]
            if acoustic:
                masked_proposal = np.ma.array(proposalsXDoc[doc], mask=Xs_mask[s:e])
                proposalsXDoc[doc] = masked_proposal.compressed()

        print('Scoring network predictions')
        if gold:
            targetsXDoc=goldEval
        else:
            if acoustic:
                for doc in targetsXDoc:
                    s, e = doc_indices[doc]
                    masked_target = np.ma.array(Y[s:e], mask=Xs_mask[s:e])
                    targetsXDoc[doc] = masked_target.compressed()
                targetsXDoc = segs2Intervals(targetsXDoc, vadIntervals)
                proposalsXDoc = segs2Intervals(proposalsXDoc, vadIntervals)
            else:
                for doc in targetsXDoc:
                    s, e = doc_indices[doc]
                    targetsXDoc[doc] = charSeq2WrdSeq(Y[s:e], goldEval[doc])
        scores = getSegScores(targetsXDoc, proposalsXDoc, acoustic=acoustic)

        print('')
        print('Segmentation score')
        printSegScores(scores, acoustic=acoustic)

        prefix = 'fixedseg'
        if segLevel:
            prefix += segLevel

        segmenter.plot(Xs[utt_ids],
                       Xs_mask[utt_ids],
                       Y[utt_ids],
                       logdir,
                       prefix,
                       i+1,
                       batch_size=batch_size)

        t1 = time.time()

        print('Iteration time: %.2fs' % (t1 - t0))
        print()


##################################################################################
##################################################################################
##
##  TEST NETWORK(S) ON FIXED SEGMENTATION
##
##################################################################################
##################################################################################

def testUnits(Xs, Xs_mask, pSegs, maxChar, maxUtt, maxLen, doc_indices, utt_ids, logdir, reverseUtt = False,
              batch_size=128, nResample=None, trainIters=100, supervisedAE=False, supervisedAEPhon=False,
              supervisedSegmenter=False, ae=None, segmenter=None, vadSegs=None, vadSegsXUtt=None, vadIntervals=None,
              ctable=None, gold=None, goldEval=None, segLevel=None, acoustic=False, fitParts=True, fitFull=False,
              vadRegionsAsUtts=False, debug=False):
    if supervisedAE:
        testFullAEOnFixedSeg(ae,
                             Xs,
                             Xs_mask,
                             pSegs,
                             maxChar,
                             maxUtt,
                             maxLen,
                             doc_indices,
                             utt_ids,
                             logdir,
                             reverseUtt=reverseUtt,
                             batch_size=batch_size,
                             nResample=nResample,
                             trainIters=trainIters,
                             vadSegs=vadSegs,
                             ctable=ctable,
                             gold=gold,
                             segLevel=segLevel,
                             acoustic=acoustic,
                             fitParts=fitParts,
                             fitFull=fitFull,
                             vadRegionsAsUtts=vadRegionsAsUtts,
                             debug=debug)

    if supervisedAEPhon:
        testPhonAEOnFixedSeg(ae,
                             Xs,
                             Xs_mask,
                             pSegs,
                             maxChar,
                             maxLen,
                             doc_indices,
                             utt_ids,
                             logdir,
                             reverseUtt=reverseUtt,
                             batch_size=batch_size,
                             nResample=nResample,
                             trainIters=trainIters,
                             vadSegs=vadSegs,
                             ctable=ctable,
                             gold=gold,
                             segLevel=segLevel,
                             acoustic=acoustic,
                             vadRegionsAsUtts=vadRegionsAsUtts,
                             debug=debug)

    if supervisedSegmenter:
        testSegmenterOnFixedSeg(segmenter,
                                Xs,
                                Xs_mask,
                                pSegs,
                                maxChar,
                                doc_indices,
                                utt_ids,
                                logdir,
                                trainIters=trainIters,
                                batch_size=batch_size,
                                vadSegs=vadSegs,
                                vadSegsXUtt=vadSegsXUtt,
                                vadIntervals=vadIntervals,
                                gold=gold,
                                goldEval=goldEval,
                                segLevel=segLevel,
                                acoustic=acoustic,
                                vadRegionsAsUtts=vadRegionsAsUtts,
                                debug=debug)