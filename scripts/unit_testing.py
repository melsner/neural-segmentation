from __future__ import print_function, division
from data_handling import *
from sampling import *
from network import *
from plotting import *
import time

def testFullAEOnFixedSeg(ae_full, ae_phon, ae_utt, embed_words, Xs, Xs_mask, pSegs, maxChar, maxUtt, maxLen, batchSize,
                         reverseUtt, nResample, trainIters, doc_indices, utt_ids, logdir, vadBreaks=None, ctable=None,
                         gold=None, segLevel=None, acoustic=False, fitParts=True, fitFull=False, debug=False):
    assert acoustic or ctable is not None, 'ctable object must be provided to testFullAEOnFixedSeg() in character mode'
    assert not acoustic or vadBreaks is not None, 'vadBreaks object must be provided to testFullAEOnFixedSeg() in acoustic mode'

    if gold is not None:
        if acoustic:
            _, goldseg = timeSegs2frameSegs(gold)
            Y = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
        else:
            Y = texts2Segs(gold, maxChar)
    else:
        Y = sampleSeg(pSegs)

    print()
    print('Unit testing full auto-encoder on fixed %s segmentation' % ('gold' if gold != None else 'random'))
    ## Re-initialize network weights in case any training has already happened
    ae_full.load_weights(logdir + '/model_init.h5', by_name=True)

    logfile = logdir + '/log_fixedfull_' + segLevel + '.txt'
    headers = ['Iteration', 'AEFullLoss']
    if not acoustic:
        headers += ['AEFullAcc']
    with open(logfile, 'wb') as f:
        print('\t'.join(headers), file=f)

    for i in range(trainIters):
        t0 = time.time()
        print('Iteration %d' % (i + 1))
        Xae, _, _, eval = trainAEOnly(ae_full,
                                      ae_phon,
                                      ae_utt,
                                      embed_words,
                                      Xs,
                                      Xs_mask,
                                      Y,
                                      maxUtt,
                                      maxLen,
                                      1,
                                      batchSize,
                                      reverseUtt,
                                      nResample=nResample,
                                      fitParts=fitParts,
                                      fitFull=fitFull)

        if nResample:
            Xae_full, _, _ = XsSeg2Xae(Xs,
                                       Xs_mask,
                                       Y,
                                       maxUtt,
                                       maxLen,
                                       nResample=None)
        prefix = 'fixedfull'
        if segLevel:
            prefix += segLevel

        plotPredsUtt(utt_ids,
                     ae_full,
                     Xae_full if nResample else Xae,
                     getYae(Xae, reverseUtt),
                     logdir,
                     prefix,
                     i+1,
                     batchSize,
                     Xae_resamp=Xae if nResample else None,
                     debug=debug)

        if not acoustic:
            print()
            print('Example reconstructions')
            printReconstruction(utt_ids, ae_full, Xae, ctable, batchSize, reverseUtt)

        row = [str(i), str(eval[0])]
        if not acoustic:
            row += [str(eval[1])]
        with open(logfile, 'ab') as f:
            print('\t'.join(row), file=f)

        t1 = time.time()
        print('Iteration time: %.2fs' %(t1-t0))
        print()


def testPhonAEOnFixedSeg(ae_full, ae_phon, Xs, Xs_mask, pSegs, maxChar, maxLen, batchSize, reverseUtt, nResample,
                         trainIters, doc_indices, utt_ids, logdir, vadBreaks=None, ctable=None, gold=None, segLevel=None,
                         acoustic=False, debug=False):
    if gold is not None:
        if acoustic:
            _, goldseg = timeSegs2frameSegs(gold)
            Y = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
        else:
            Y = texts2Segs(gold, maxChar)
    else:
        Y = sampleSeg(pSegs)

    print()
    print('Unit testing phonological auto-encoder on fixed %s segmentation' % ('gold' if gold is not None else 'random'))
    ## Re-initialize network weights in case any training has already happened
    ae_full.load_weights(logdir + '/model_init.h5', by_name=True)

    for i in range(trainIters):
        t0 = time.time()
        print('Iteration %d' % (i + 1))
        Xae = trainAEPhonOnly(ae_phon,
                              Xs,
                              Xs_mask,
                              Y,
                              maxLen,
                              1,
                              batchSize,
                              reverseUtt,
                              nResample)

        if reverseUtt:
            Yae = np.flip(Xae, 1)
        else:
            Yae = Xae

        if nResample:
            Xae_full, _, _ = XsSeg2XaePhon(Xs,
                                           Xs_mask,
                                           Y,
                                           maxLen,
                                           nResample=None)

        prefix = 'fixedphon'
        if segLevel:
            prefix += segLevel

        plotPredsWrd(utt_ids,
                     ae_phon,
                     Xae_full if nResample else Xae,
                     getYae(Xae, reverseUtt),
                     logdir,
                     prefix,
                     i+1,
                     batchSize,
                     Xae_resamp=Xae if nResample else None,
                     debug=debug)

        if not acoustic:
            print()
            print('Example reconstructions')
            printReconstruction(utt_ids, ae_phon, Xae, ctable, batchSize, reverseUtt)

        t1 = time.time()
        print('Iteration time: %.2fs' %(t1-t0))
        print()

def testSegmenterOnFixedSeg(segmenter, Xs, Xs_mask, pSegs, maxChar, batchSize, segShift, trainIters, doc_indices,
                            utt_ids, logdir, vadBreaks=None, vad=None, intervals=None, gold=None, goldEval=None,
                            segLevel=None, acoustic=False, debug=False):
    if gold:
        if acoustic:
            _, goldseg = timeSegs2frameSegs(gold)
            Y = frameSegs2FrameSegsXUtt(goldseg, vadBreaks, maxChar, doc_indices)
        else:
            Y = texts2Segs(gold, maxChar)
    else:
        Y = sampleSeg(pSegs)

    print()
    print('Unit testing segmenter network on fixed %s segmentation' % ('gold' if gold is not None else 'random'))

    proposalsXDoc = dict.fromkeys(doc_indices.keys())
    targetsXDoc = dict.fromkeys(doc_indices)

    ## Re-initialize network weights in case any training has already happened
    segmenter.load_weights(logdir + '/segmenter_init.h5', by_name=True)

    segsProposal = trainSegmenterOnly(segmenter,
                                      Xs,
                                      Xs_mask,
                                      Y,
                                      trainIters,
                                      batchSize,
                                      segShift)

    if acoustic:
        segsProposal[np.where(vad)] = 1.
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
            targetsXDoc = frameSegs2timeSegs(intervals, targetsXDoc)
            proposalsXDoc = frameSegs2timeSegs(intervals, proposalsXDoc)
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

    plotPredsSeg(utt_ids,
                 segmenter,
                 Xs,
                 Xs_mask,
                 Y,
                 logdir,
                 prefix,
                 1,
                 segShift,
                 batchSize,
                 debug)

def testUnits(Xs, Xs_mask, pSegs, maxChar, maxUtt, maxLen, batchSize, reverseUtt, nResample, segShift, trainIters,
              doc_indices, utt_ids, logdir, supervisedAE=False, supervisedAEPhon=False, supervisedSegmenter=False,
              ae_full=None, ae_phon=None, ae_utt=None, embed_words=None, segmenter=None, vadBreaks=None, vad=None,
              intervals=None, ctable=None, gold=None, goldEval=None, segLevel=None, acoustic=False, fitParts=True,
              fitFull=False, debug=False):
    if supervisedAE:
        testFullAEOnFixedSeg(ae_full,
                             ae_phon,
                             ae_utt,
                             embed_words,
                             Xs,
                             Xs_mask,
                             pSegs,
                             maxChar,
                             maxUtt,
                             maxLen,
                             batchSize,
                             reverseUtt,
                             nResample,
                             trainIters,
                             doc_indices,
                             utt_ids,
                             logdir,
                             vadBreaks=vadBreaks,
                             ctable=ctable,
                             gold=gold,
                             segLevel=segLevel,
                             acoustic=acoustic,
                             fitParts=fitParts,
                             fitFull=fitFull,
                             debug=debug)

    if supervisedAEPhon:
        testPhonAEOnFixedSeg(ae_full,
                             ae_phon,
                             Xs,
                             Xs_mask,
                             pSegs,
                             maxChar,
                             maxLen,
                             batchSize,
                             reverseUtt,
                             nResample,
                             trainIters,
                             doc_indices,
                             utt_ids,
                             logdir,
                             vadBreaks=vadBreaks,
                             ctable=ctable,
                             gold=gold,
                             segLevel=segLevel,
                             acoustic=acoustic,
                             debug=debug)

    if supervisedSegmenter:
        testSegmenterOnFixedSeg(segmenter,
                                Xs,
                                Xs_mask,
                                pSegs,
                                maxChar,
                                batchSize,
                                segShift,
                                trainIters,
                                doc_indices,
                                utt_ids,
                                logdir,
                                vadBreaks=vadBreaks,
                                vad=vad,
                                intervals=intervals,
                                gold=gold,
                                goldEval=goldEval,
                                segLevel=segLevel,
                                acoustic=acoustic,
                                debug=debug)