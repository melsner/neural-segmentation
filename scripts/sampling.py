import sys, numpy as np
from numpy import ma

def sampleSeg(pSegs):
    smp = np.random.uniform(size=pSegs.shape)
    return smp < pSegs

def sampleSegs(pSegs):
    segs = dict.fromkeys(pSegs)
    for doc in pSegs:
        segs[doc] = sampleSeg(pSegs[doc])
    return segs

def lossByUtt(model, Xb, yb, BATCH_SIZE, metric="logprob"):
    preds = ma.array(model.predict(Xb, batch_size=BATCH_SIZE, verbose=0), mask=ma.getmask(yb))

    if metric == "logprob":
        logP = np.log(preds)
        pRight = logP * yb
        # Sum out char, len(chars)
        return pRight.sum(axis=(2, 3))
    elif metric in ['mse', 'mse1best']:
        se = (preds - yb) ** 2
        # Sum out char, len(chars)
        mse = np.mean(se, axis=(2, 3))
        return mse
    else:
        raise ValueError('''The loss metric you have requested ("%s") is not supported.
                            Supported metrics are "logprob" and "mse".''')


def guessSegTargets(scores, segs, priorSeg, algorithm='importance', verbose=True):
    if algorithm in ['importance', '1best']:
        scores = np.array(scores)
        scores = scores.sum(-1)
        MM = np.max(scores.sum(-1), axis=0, keepdims=True)
        eScores = np.exp(scores - MM)
        # approximately the probability of the sample given the data
        samplePrior = eScores / eScores.sum(axis=1, keepdims=True)
        bestSampleXUtt = np.argmax(samplePrior, axis=1)
        bestSegXUtt = []
        bestSampleScore = 0
        bestSamplePrior = 0
        for ix in xrange(len(bestSampleXUtt)):
            bestSegXUtt.append(segs[ix][bestSampleXUtt[ix], :])
            bestSampleScore += scores[ix, bestSampleXUtt[ix]]
            bestSamplePrior += samplePrior[ix, bestSampleXUtt[ix]]
        bestSamplePrior /= len(bestSampleXUtt)
        bestSegXUtt = np.array(bestSegXUtt)
        if verbose:
            print('      Best segmentation set: score = %s, prob = %s, num segs = %s' % (
            bestSampleScore, bestSamplePrior, bestSegXUtt.sum()))

        if algorithm == '1best':
            return bestSegXUtt, bestSegXUtt

        # proposal prob
        priorSeg = np.expand_dims(priorSeg, 1)
        qSeg = segs * priorSeg + (1 - segs) * (1 - priorSeg)

        wts = np.expand_dims(samplePrior, -1) / qSeg
        wts = wts / wts.sum(axis=1, keepdims=True)

        # print("score distr:", dist[:10])

        # print("shape of segment matrix", segmat.shape)
        # print("best score sample", np.argmax(-scores))
        # print("transformed losses for utt 0", eScores[:, 0])
        # print("best score sample", np.argmax(eScores))
        # best_score = np.argmax(eScores)

        # print("top row of distr", pSeg[:, 0])
        # print("top row of correction", qSeg[:, 0])
        # print("top row of weights", wts[:, 0])

        # sample x utterance x segment
        # nSamples = segmat.shape[1]
        wtSegs = segs * wts

        # for si in range(nSamples):
        #    print("seg vector", si, segmat[si, 0, :])
        #    print("est posterior", pSeg[si, 0])
        #    print("q", qSeg[si, 0])
        #    print("weight", wts[si, 0])
        #    print("contrib", wtSegs[si, 0])

        segWts = np.expand_dims(wtSegs.sum(axis=1), -1)
        bestSampleXUtt = segWts > .5

        # print("        Difference between best sample and guessed best", np.sum(segs[best_score]-best))
        # print("        Total segmentations (best sample)", np.sum(segs[best_score]))
        # print("        Total segmentations (best guess)", np.sum(best))

        # print("top row of wt segs", segWts[0])
        # print("max segs", best[0])
    else:
        raise ValueError('''The sampling algorithm you have requested ("%s") is not supported.'
                            Please use one of the following:

                            importance
                            1best''')
    return segWts, bestSampleXUtt


def KL(pSeg1, pSeg2):
    return np.mean(pSeg1 * (np.log(pSeg1 / pSeg2)) + (1 - pSeg1) * (np.log((1 - pSeg1) / (1 - pSeg2))))

