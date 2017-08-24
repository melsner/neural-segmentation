import sys, numpy as np
from numpy import ma, inf, nan
from data_handling import XsSeg2Xae, getYae, pSegs2Segs
from network import sparse_xent




##################################################################################
##################################################################################
##
##  UTILITY METHODS
##
##################################################################################
##################################################################################

def getRandomPermutation(n):
    p = np.random.permutation(np.arange(n))
    p_inv = np.zeros_like(p)
    p_inv[p] = np.arange(n)
    return p, p_inv





##################################################################################
##################################################################################
##
##  METHODS FOR PROCESSING SAMPLED SEGMENTATIONS
##
##################################################################################
##################################################################################

def scoreXUtt(ae, Xs, Xs_mask, segs, maxUtt, maxLen, logdir,
              reverseUtt=False, batch_size=128, nResample=None, agg='mean', train=False):
    if train:
        ae.save(logdir, suffix='_tmp')
        Xae, deletedChars, oneLetter, eval = ae.update(Xs, Xs_mask, segs, maxUtt, maxLen, verbose=False)
    else:
        Xae, deletedChars, oneLetter = XsSeg2Xae(Xs,
                                                 Xs_mask,
                                                 segs,
                                                 maxUtt,
                                                 maxLen,
                                                 nResample)

    Yae = getYae(Xae, reverseUtt)

    if train:
        ae.load(logdir, suffix='_tmp')

    score = -ae.loss_tensor([Xae, Yae], batch_size=batch_size)
    score = score * Yae.any(-1)
    if agg == 'mean':
        score[score == 0] = nan
        with np.errstate(divide='ignore'):
            score = np.nan_to_num(np.nanmean(score, axis=-1))
    elif agg== 'sum':
        score = score.sum(-1)
    else:
        raise ValueError('''The aggregation function you have requested ("%s") is not supported.
                            Supported aggregators are "mean" and "sum".''' % agg)
    return score, Xae, deletedChars, oneLetter


def importanceWeights(scores, segs, Xs_mask, annealer, local=True):
    if local:
        ## Returns a tensor of shape (nUtt, nSample, nChar) with local importance weights by character
        scores_infpad = scores.copy()
        scores_infpad[scores_infpad==0] = -inf
        MM = np.max(scores_infpad, axis=(1,2), keepdims=True)
        scores -= MM
        if annealer is not None:
            print('Score annealing temperature: %.4f' % annealer.temp())
            scores *= float(1)/annealer.step()
        scores = np.exp(scores)
        ## Get indices of words spanning each time step in each sample
        wix = segs.cumsum(-1).astype('int')
        ## Get offsets for flattened scores array
        offset = np.arange(0, scores.shape[0]*scores.shape[1]*scores.shape[2], scores.shape[2]).reshape((scores.shape[0],scores.shape[1],1))
        ## Get index of first non-padding word per sample
        padding = np.clip(scores.shape[2] - 1 - np.expand_dims(wix[...,-1], -1), 0, inf).astype('int')
        ## Get score of word loss corresponding to each timestep
        wts = np.take(scores, np.clip(wix,0,scores.shape[-1]-1)+offset+padding)
        wts[wix > scores.shape[-1]] = 0
    else:
        ## Returns a tensor of shape (nUtt, nSample, 1) with utterance-wide importance weights by utterance
        scores[scores==0] = nan
        scores = np.nanmean(scores, axis=-1)
        MM = np.max(scores, axis=1, keepdims=True)
        scores -= MM
        if annealer is not None:
            print('Score annealing temperature: %.4f' % annealer.temp())
            scores *= float(1)/annealer.step()
        scores = np.exp(scores)
        wts = np.expand_dims(scores / scores.sum(-1, keepdims=True), -1)
    return wts

def oneBest(scores, annealer, segs):
    scores[scores == 0] = nan
    scores = np.nanmean(scores, axis=-1)
    oldscores = scores.copy()
    MM = scores.max(-1, keepdims=True)
    scores -= MM
    if annealer is not None:
        scores *= float(1) / annealer.temp()
    scores = np.exp(scores)
    scores /= scores.sum(-1, keepdims=True)
    best = scores.argmax(-1)
    oneBestSegs = segs[np.arange(len(segs)), best]
    print('One-best segmentation set: score = %.4f, num segs = %d' % (oldscores[np.arange(len(oldscores)), best].mean(), oneBestSegs.sum()))
    return oneBestSegs

def lossXChar(model, Xae, Yae, batch_size, acoustic=False, loss='xent'):
    preds = model.predict(Xae, batch_size=batch_size)

    if acoustic:
        # MSE loss
        real_chars = Yae.any(-1, keepdims=True)
    else:
        real_chars = (np.expand_dims(Yae.argmax(-1), -1) > 0).any(-1, keepdims=True)

    if loss == 'xent':
        # Cross-entropy loss
        with np.errstate(divide='ignore'):
            loss = - np.nan_to_num(np.log(preds)) * Yae
        ## Zero-out scores for padding chars
        loss *= real_chars
    elif loss == 'binary-xent':
        # Binary cross-entropy loss
        with np.errstate(divide='ignore'):
            loss = - (np.nan_to_num(np.log(preds))*Yae + np.nan_to_num(np.log(1-preds))*(1-Yae))
        loss *= real_chars
    elif loss == 'mse':
        # MSE loss
        loss = (Yae - preds) ** 2
        ## Zero-out scores for padding chars
        loss *= real_chars
        loss = loss.mean(-1)
    else:
        raise ValueError('''The loss metric you have requested ("%s") is not supported.
                                    Supported metrics are "logprob" and "mse".''' %loss)

    loss = loss.sum() / real_chars.sum()

    return loss

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
        self.uttLenChar = None
        self.bestpath = None
        self.bestscore = None

    def __str__(self):
        string = 'Nodes:\n'
        for n in self.nodes:
            string += str(self.nodes[n]) + '\n'
        string += '\nEdges:\n'
        for e in self.edges:
            string += str(self.edges[e]) + '\n'
        if self.bestpath:
            string += '\nBest path:\n'
            for n in self.bestpath:
                string += str(n) + '\n'
        if self.bestscore is not None:
            string += '\nBest score: %.4f\n' %self.bestscore
        return string

    def addNode(self, key):
        if key not in self.nodes:
            self.nodes[key] = Node(key)

    def addEdge(self, a, b, wt):
        if not isinstance(a, Node):
            a = self.nodes[a]
        if not isinstance(b, Node):
            b = self.nodes[b]
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

    def score(self, score, wLen, maxLen, delWt, oneLetterWt, segWt):
        score -= max(0, wLen-maxLen) * delWt
        score -= (wLen==1) * oneLetterWt
        # if wLen > 0:
        #     ## Penalize segmentations by adding setWt additional characters to the loss
        #     segPenalty = score / wLen
        #     score -= segWt*segPenalty
        #     print(segPenalty)
        #     raw_input()
        score -= segWt
        return score

    def buildFromSegs(self, segs, wts, seg_mask=None, maxLen=inf, delWt=0, oneLetterWt=0, segWt=0):
        assert len(segs) == len(wts), 'Segmentation and score matrices have different numbers of samples (%d and %d respectively)' %(len(segs), len(wts))
        if seg_mask is not None:
            self.uttLenChar = segs.shape[1] - seg_mask.sum()
        self.addNode('start')
        self.addNode('end')
        for s in range(len(segs)):
            last = 'start'
            seg = segs[s]
            wt = wts[s]
            wt = wt[wt < 0]
            uttLenWrd = len(wt)
            if segs[s,0] == 1:
                self.addNode(0)
                self.addEdge('start', 0, 0)
                last = 0
                t = 1
            else:
                t = 0
            w = 0
            assert uttLenWrd <= 64, str(uttLenWrd)
            while t < self.uttLenChar and w < uttLenWrd:
                if seg[t] == 1:
                    self.addNode(t)
                    wLen = t-last if last != 'start' else t
                    score = self.score(wt[w], wLen, maxLen, delWt, oneLetterWt, segWt)
                    self.addEdge(last, t, score)
                    last = t
                    w += 1
                t += 1
            t = self.uttLenChar
            wLen = t - last if last != 'start' else t
            score = self.score(wt[w] if w < uttLenWrd else delWt*min(wLen,maxLen), wLen, maxLen, delWt, oneLetterWt, segWt)
            self.addEdge(last, 'end', score)

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
        bestscore = u.score
        while u.prev != None:
            u = u.prev
            if u.t != 'start':
                segseq.insert(0, u.t)
        self.bestpath = segseq
        self.bestscore = bestscore
        return segseq, bestscore

def viterbiDecode(segs, scores, Xs_mask, maxLen, delWt, oneLetterWt, segWt):
    l = Lattice()
    l.buildFromSegs(segs, scores, Xs_mask, maxLen, delWt, oneLetterWt, segWt)
    segseq, finalscore = l.dijkstra()
    segsOut = np.zeros((segs.shape[1]))
    finalscore /= l.uttLenChar
    segsOut[segseq] = 1

    return segsOut, finalscore

def guessSegTargets(scores, penalties, segs, priorSeg, Xs_mask, algorithm='viterbi', maxLen=inf, delWt=0, oneLetterWt=0,
                    segWt=0, annealer=None, importanceSampledSegTargets=False, acoustic=False, verbose=True):
    augscores = scores + penalties

    if algorithm.startswith('importance') or importanceSampledSegTargets:
        ## Extract importance weights
        wts = importanceWeights(augscores, segs, Xs_mask, annealer=annealer, local=algorithm.endswith('Local'))

        ## Compute correction distribution
        priorSeg = np.expand_dims(np.squeeze(priorSeg, -1), 1)
        qSeg = segs * priorSeg + (1 - segs) * (1 - priorSeg)
        wts = wts / qSeg
        with np.errstate(divide='ignore'):
            wts = np.nan_to_num(wts / wts.sum(axis=1, keepdims=True))

        ## Multiply in segmentation decisions
        wtSegs = segs * wts
        segTargetsXUtt = np.expand_dims(wtSegs.sum(axis=1), -1)

    if algorithm == '1best':
        bestSegXUtt = oneBest(augscores, annealer, segs)
        if not importanceSampledSegTargets:
            segTargetsXUtt = np.expand_dims(bestSegXUtt, -1)
    elif algorithm.startswith('importance'):
        bestSegXUtt = pSegs2Segs(segTargetsXUtt, acoustic)
        nSeg = bestSegXUtt.sum()
        print('Importance-sampled segmentation set: num segs = %d' % nSeg)
    elif algorithm == 'viterbi':
        batch_score = 0
        bestSegXUtt = np.zeros((priorSeg.shape[0], priorSeg.shape[1]))
        for i in range(len(scores)):
            bestSegXUtt[i], utt_score = viterbiDecode(segs[i],
                                                      scores[i],
                                                      Xs_mask[i],
                                                      maxLen,
                                                      delWt,
                                                      oneLetterWt,
                                                      segWt)
            batch_score += utt_score
        batch_score /= len(scores)
        bestSegXUtt = np.expand_dims(bestSegXUtt, -1)
        print('Best Viterbi-decoded segmentation set: score = %.4f, num segs = %d' %(batch_score, bestSegXUtt.sum()))
        if not importanceSampledSegTargets:
            segTargetsXUtt = bestSegXUtt
    else:
        raise ValueError('''The sampling algorithm you have requested ("%s") is not supported.'
                            Please use one of the following:

                            importance
                            1best
                            viterbi''')
    return segTargetsXUtt, bestSegXUtt


def KL(pSeg1, pSeg2):
    return np.mean(pSeg1 * (np.log(pSeg1 / pSeg2)) + (1 - pSeg1) * (np.log((1 - pSeg1) / (1 - pSeg2))))