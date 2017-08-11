import sys, numpy as np
from numpy import ma, inf, nan
from scipy.signal import argrelmax, argrelmin





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
##  METHODS FOR EXTRACTING SEGMENTATIONS
##
##################################################################################
##################################################################################

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





##################################################################################
##################################################################################
##
##  METHODS FOR PROCESSING SAMPLED SEGMENTATIONS
##
##################################################################################
##################################################################################

def scoreXWrd2ScoreXChar(scores, segs, annealer, local=True):
    if local:
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
        charscores = np.take(scores, np.clip(wix,0,scores.shape[-1]-1)+offset+padding)
        charscores[wix > scores.shape[-1]] = 0
        # ix = 127
        # print('utt %d' %ix)
        # print('scores')
        # print(scores[ix])
        # print('segs')
        # print(segs[ix])
        # print('wix')
        # print(wix[ix])
        # print('offset')
        # print(offset[ix])
        # print('padding')
        # print(padding[ix])
        # print('charscores')
        # print(charscores.shape)
        # print(charscores[ix])
        # raw_input()
    else:
        scores = scores.sum(-1)
        MM = np.max(scores, axis=1, keepdims=True)
        scores -= MM
        if annealer is not None:
            print('Score annealing temperature: %.4f' % annealer.temp())
            scores *= float(1)/annealer.step()
        scores = np.exp(scores)
        charscores = np.expand_dims(scores / scores.sum(-1, keepdims=True), -1)
    return charscores

def oneBest(scores, annealer, segs):
    eScores = scores.sum(-1)
    MM = eScores.max(-1, keepdims=True)
    eScores -= MM
    if annealer is not None:
        eScores *= float(1) / annealer.temp()
    eScores = np.exp(eScores)
    eScores /= eScores.sum(-1, keepdims=True)
    best = eScores.argmax(-1)
    oneBestSegs = segs[np.arange(len(segs)), best]
    print('One-best segmentation score: %.4f' % scores[np.arange(len(scores)), best].sum())
    print('One-best mean sample prob: %.4f' % eScores[np.arange(len(eScores)), best].mean())
    print('One-best segmentation count: %d' % oneBestSegs.sum())
    return oneBestSegs

def scoreXUtt(model, Xae, Yae, batch_size, reverseUtt, metric="logprob", debug=False):
    preds = model.predict(Xae, batch_size=batch_size)

    if metric == "logprob":
        score = np.nan_to_num(np.log(preds))
        score = score * Yae
        #print(Yae.argmax(-1))
        #print(score)
        ## Zero-out scores for padding chars
        score = score * (np.expand_dims(Yae.argmax(-1), -1) > 0).any(-1, keepdims=True)
        ## Sum out char, len(chars)
        score = score.sum(axis=(2,3))
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
            score = np.flip(score, 1)
    elif metric == 'logprobbinary':
        score = np.nan_to_num(np.log(preds)) * Yae + np.nan_to_num(np.log(1-preds)) * (1-Yae)
        ## Zero-out scores for padding chars
        score = score * (np.expand_dims(Xae.argmax(-1), -1) > 0).any(-1, keepdims=True)
        score = np.expand_dims(score.sum(axis=(1,2)), -1)
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
            score = np.flip(score, 1)
    else:
        raise ValueError('''The loss metric you have requested ("%s") is not supported.
                            Supported metrics are "logprob" and "mse".''' %metric)

    ## Zero out losses from padding regions
    return score

def lossXChar(model, Xae, Yae, batch_size, acoustic=False, loss='xent'):
    preds = model.predict(Xae, batch_size=batch_size)

    if acoustic:
        # MSE loss
        real_chars = Yae.any(-1, keepdims=True)
    else:
        real_chars = (np.expand_dims(Yae.argmax(-1), -1) > 0).any(-1, keepdims=True)

    if loss == 'xent':
        # Cross-entropy loss
        loss = - np.nan_to_num(np.log(preds)) * Yae
        ## Zero-out scores for padding chars
        loss *= real_chars
    elif loss == 'binary-xent':
        # Binary cross-entropy loss
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

    ## Construct lattice
    lattice = Lattice()
    lattice.addNode('start')
    lattice.addNode('end')

    for s in range(segs.shape[0]):
        src = lattice.nodes['start']
        src_t = -inf
        w = -1
        ## Scan to get the index of the first real word score
        while w < scores.shape[1]-1 and scores[s,w+1] == 0.:
            w += 1
        w += int(segs[s][0] == 0)
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
                    wt = getViterbiWordScore(wt, wLen, maxLen, delWt, oneLetterWt, segWt) # * wLen/uttLen # scale loss by character
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

    ## Shortest path
    segsOut = np.zeros((segs.shape[1]))
    segseq, finalscore = lattice.dijkstra()
    segsOut[segseq] = 1

    # print('')
    # print(segs)
    # print(scores)
    # print(lattice)
    # print(segseq)
    # print(segsOut)
    # print(finalscore)
    # raw_input()

    return segsOut, finalscore

def guessSegTargets(scores, penalties, segs, priorSeg, Xs_mask, algorithm='viterbi', maxLen=inf, delWt=0, oneLetterWt=0,
                    segWt=0, annealer=None, importanceSampledSegTargets=False, acoustic=False, verbose=True):
    augscores = scores + penalties

    if algorithm.startswith('importance') or importanceSampledSegTargets:
        priorSeg = np.expand_dims(np.squeeze(priorSeg, -1), 1)
        qSeg = segs * priorSeg + (1 - segs) * (1 - priorSeg)

        charScores = scoreXWrd2ScoreXChar(augscores, segs, annealer=annealer, local=algorithm.endswith('Local'))
        wts = charScores / qSeg
        wts = np.nan_to_num(wts / wts.sum(axis=1, keepdims=True))
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
        bestSegXUtt = np.expand_dims(bestSegXUtt, -1)
        print('Best Viterbi-decoded segmentation set: score = %s, num segs = %d' %(batch_score, bestSegXUtt.sum()))
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