import sys, numpy as np
from numpy import ma, inf, nan
from data_handling import getYae

def getRandomPermutation(n):
    p = np.random.permutation(np.arange(n))
    p_inv = np.zeros_like(p)
    p_inv[p] = np.arange(n)
    return p, p_inv

def sampleSeg(pSegs):
    smp = np.random.uniform(size=pSegs.shape)
    return smp < pSegs

def sampleSegs(pSegs):
    segs = dict.fromkeys(pSegs)
    for doc in pSegs:
        segs[doc] = sampleSeg(pSegs[doc])
    return segs

def scoreXUtt(model, Xae, Yae, batch_size, reverseUtt, metric="logprob", debug=False):
    preds = model.predict(np.concatenate(Xae, 0), batch_size=batch_size)
    preds = np.stack(np.split(preds, Xae.shape[0]))

    if metric == "logprob":
        score = np.log(preds)
        score = score * Yae
        #print(Yae.argmax(-1))
        #print(score)
        ## Zero-out scores for padding chars
        score = score * (np.expand_dims(Yae.argmax(-1), -1) > 0).any(-1, keepdims=True)
        ## Sum out char, len(chars)
        score = score.sum(axis=(-2, -1))
        # if debug:
        #     for u in range(len(score1)):
        #         for w in range(len(score1[u])):
        #             print(score1[u, w])
        #             print(score2[u, w])
        #             print(score3[u, w])
        #             print((np.expand_dims(Yae[u, w].argmax(-1), -1) > 0).sum())
        #             print(score4[u,w])
        #             raw_input('Press any key to continue')
        if reverseUtt:
            score = np.flip(score, 2)
    elif metric == 'mse':
        ## Initialize score as negative squared error
        score = -((preds - Yae) ** 2)
        ## Zero-out scores for padding chars
        score *= Yae.any(-1, keepdims=True)
        ## Get MSE per character
        score = score.mean(-1)
        ## Sum MSE's within each word
        score = score.sum(-1)
        if reverseUtt:
            score = np.flip(score, 2)
    else:
        raise ValueError('''The loss metric you have requested ("%s") is not supported.
                            Supported metrics are "logprob" and "mse".''')

    ## Zero out losses from padding regions
    #print(score)
    #raw_input()
    return score

class Node(object):
    def __init__(self, t=None):
        self.t = t
        self.outgoing = set()
        self.score = -inf
        self.prev = None

    def __str__(self):
        return str(self.t) + ' (prev: ' + str(self.prev.t if self.prev else self.prev) + ', score: ' + str(self.score) + ')'

class Edge(object):
    def __init__(self, a, b, wt):
        self.a = a
        self.b = b
        self.wt = wt

    def __str__(self):
        return str(self.a.t) + '->' + str(self.b.t) + ' (' + str(self.wt) + ')'

class Lattice(object):
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def __str__(self):
        string = 'Nodes:\n'
        for n in self.nodes:
            string += str(self.nodes[n]) + '\n'
        string += '\nEdges:\n'
        for e in self.edges:
            string += str(self.edges[e]) + '\n'
        return string

    def addNode(self, key):
        if key not in self.nodes:
            self.nodes[key] = Node(key)

    def addEdge(self, a, b, wt):
        key = (a,b)
        if key in self.edges:
            if self.edges[key].wt < wt:
                self.edges[key].wt = wt
        else:
            self.edges[key] = Edge(a, b, wt)
            a.outgoing.add(b)

    def length(self, a, b):
        if a == b:
            return 0
        return self.edges[(a,b)].wt

    def dijkstra(self):
        Q = set(self.nodes.values())
        u = self.nodes['start']
        u.score = 0
        while len(Q) > 0:
            maxQ = -inf
            argmaxQ = None
            for x in Q:
                if x.score > maxQ:
                    maxQ = x.score
                    argmaxQ = x
            u = argmaxQ
            if u == None:
                Q = set(self.nodes.values())
                for x in Q:
                    print(self)
                    print(x.t)
                    print(x.prev)
                    print(x.score)
                    print('')
            Q.remove(u)
            if u.t == 'end':
                break
            for v in u.outgoing:
                score = u.score + self.length(u,v)
                if score > v.score:
                    v.score = score
                    v.prev = u
        segseq = []
        #print(self)
        finalscore = u.score
        #print(finalscore)
        while u.prev != None:
            u = u.prev
            if u.t != 'start':
                segseq.insert(0, u.t)
        #print(segseq)
        #raw_input()
        return segseq, finalscore

def getViterbiWordScore(score, wLen, maxLen, delWt, oneLetterWt, segWt):
    score -= max(0, wLen-maxLen) * delWt
    score -= (wLen==1) * oneLetterWt
    score -= segWt
    return score

def viterbiDecode(segs, scores, Xs_mask, maxLen, delWt, oneLetterWt, segWt):
    uttLen = segs.shape[1] - Xs_mask.sum()
    #print(uttLen)

    ## Construct lattice
    lattice = Lattice()
    lattice.addNode('start')
    lattice.addNode('end')

    for s in range(segs.shape[0]):
        #print(np.where(segs[s]))
        #print(scores[s])
        src = lattice.nodes['start']
        src_t = -inf
        ## Set score index to -1, shift rightward if no boundary at first timestep
        w = -1 + int(not segs[s][0] == 1)
        ## Scan to get the index of the first real word score
        while w < scores.shape[1]-1 and scores[s,w+1] == 0.:
            w += 1
        #print(w)
        t = 0
        while t <= segs.shape[1]:
            if t < segs.shape[1] and w < scores.shape[1]-1:
                seg = segs[s,t]
                label = t
            else:
                t = uttLen
                seg = 1
                label = 'end'
            if seg == 1:
                lattice.addNode(label)
                dest = lattice.nodes[label]
                if w >= 0:
                    wt = scores[s, w]
                    wLen = min(t,uttLen) - max(0,src_t)
                    wt = getViterbiWordScore(wt, wLen, maxLen, delWt, oneLetterWt, segWt)
                    #print(t, wLen, src_t, wt, scores[s,w])
                    #raw_input()
                else:
                    wt = 0
                lattice.addEdge(src, dest, wt)
                w += 1
                src = dest
                src_t = src.t
                ## Handle case when we have reached maxUtt
                if src_t == 'end':
                    break
            t += 1

    #print('')
    #print(lattice)
    #print(scores)
    #print(segs)

    ## Shortest path
    segsOut = np.zeros((segs.shape[1]))
    #print(uttLen)
    segseq, finalscore = lattice.dijkstra()
    segsOut[segseq] = 1
    #print(segsOut)
    #print(finalscore)
    #raw_input('Press any key to continue')

    return np.expand_dims(segsOut, -1), finalscore

def guessSegTargets(scores, segs, priorSeg, Xs_mask, algorithm='viterbi', maxLen=inf, delWt=0, oneLetterWt=0, segWt=0, verbose=True):
    if algorithm in ['importance', '1best']:
        scores = scores.sum(-1)
        MM = np.max(scores, axis=1, keepdims=True)
        eScores = np.exp(scores - MM + 1e-5)
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
            print('Best segmentation set: score = %s, prob = %s, num segs = %s' % (
            bestSampleScore, bestSamplePrior, bestSegXUtt.sum()))

        if algorithm == '1best':
            return bestSegXUtt, bestSegXUtt ## Seg targets (1st output) and best segmentation (2nd output) are identical.

        ## Proposal prob
        priorSeg = np.expand_dims(np.squeeze(priorSeg, -1), 1)
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

        segTargetsXUtt = np.expand_dims(wtSegs.sum(axis=1), -1)
        bestSampleXUtt = segTargetsXUtt > .5

        # print("        Difference between best sample and guessed best", np.sum(segs[best_score]-best))
        # print("        Total segmentations (best sample)", np.sum(segs[best_score]))
        # print("        Total segmentations (best guess)", np.sum(best))

        # print("top row of wt segs", segWts[0])
        # print("max segs", best[0])
    elif algorithm == 'viterbi':
        bestSampleXUtt = np.zeros_like(priorSeg)
        batch_score = 0
        for i in range(len(scores)):
            bestSampleXUtt[i], utt_score = viterbiDecode(segs[i],
                                                         scores[i],
                                                         Xs_mask[i],
                                                         maxLen,
                                                         delWt,
                                                         oneLetterWt,
                                                         segWt)
            batch_score += utt_score
        print('Best segmentation set: score = %s, num segs = %s' %(batch_score, bestSampleXUtt.sum()))
        segTargetsXUtt = bestSampleXUtt
    else:
        raise ValueError('''The sampling algorithm you have requested ("%s") is not supported.'
                            Please use one of the following:

                            importance
                            1best
                            viterbi''')
    return segTargetsXUtt, bestSampleXUtt


def KL(pSeg1, pSeg2):
    return np.mean(pSeg1 * (np.log(pSeg1 / pSeg2)) + (1 - pSeg1) * (np.log((1 - pSeg1) / (1 - pSeg2))))

