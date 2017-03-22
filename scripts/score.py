from __future__ import division
import sys

def precision_recall(n_matched, n_gold, n_proposed):
    """Calculates the classification precision and recall, given
    the number of true positives, the number of existing positives,
    and the number of proposed positives."""

    if n_gold > 0:
        recall = n_matched / n_gold
    else:
        recall = 0.0

    if n_proposed > 0:
        precision = n_matched / n_proposed
    else:
        precision = 0.0

    return precision, recall

def fscore(precision, recall, beta=1.0):
    """Calculates the f-score (default is balanced f-score; beta > 1
    favors precision), the harmonic mean of precision and recall."""

    num = (beta**2 + 1) * precision * recall
    denom = (beta**2) * precision + recall
    if denom == 0:
        return 0.0
    else:
        return num / denom

def precision_recall_f(n_matched, n_gold, n_proposed, beta=1.0):
    """Calculates precision, recall and f-score."""

    prec,rec = precision_recall(n_matched, n_gold, n_proposed)
    f = fscore(prec, rec, beta=beta)

    return prec,rec,f

def getBreaks(words):
    breaks = []
    soFar = 0
    for word in words:
        soFar += len(word)
        breaks.append(soFar)
    breaks.pop(-1) #end-of-string isn't a break
    return breaks

def scoreBreaks(surface, found):
    matched = 0
    proposed = 0
    actual = 0

    warned = 0

    for wReal,wFound in zip(surface, found):
        # print "real:", wReal
        # print "found:", wFound

        if type(wReal[0]) == str:
            cat1 = "".join(wReal)
            cat2 = "".join(wFound)
        else:
            cat1 = sum(wReal, [])
            cat2 = sum(wFound, [])

        if cat1 != cat2:
            if not warned:
                print "Warning: surface string mismatch:", cat1, cat2
                warned = 1
            elif warned == 1:
                print "Warning: more mismatches"
                warned += 1
            #sys.exit(1)

        trueBreaks = set(getBreaks(wReal))

        foundBreaks = set(getBreaks(wFound))

        actual += len(trueBreaks)
        proposed += len(foundBreaks)
        matched += len(trueBreaks.intersection(foundBreaks))

    return precision_recall_f(matched, actual, proposed)

def getBounds(wordLst):
    bounds = []

    x0 = 0
    for ii,wi in enumerate(wordLst):
        (b1,b2) = (x0, x0 + len(wordLst[ii]))
        bounds.append((b1, b2))
        x0 += len(wordLst[ii])

    return bounds

def scoreWords(underlying, found):
    matched = 0
    proposed = 0
    actual = 0

    for wReal,wFound in zip(underlying, found):
        realBounds = getBounds(wReal)
        foundBounds = getBounds(wFound)

        # print set(realSeq).intersection(foundSeq)

        actual += len(realBounds)
        proposed += len(foundBounds)
        matched += len(set(realBounds).intersection(foundBounds))

    return precision_recall_f(matched, actual, proposed)

def scoreLexicon(underlying, found):
    realLex = set()
    foundLex = set()

    for wReal,wFound in zip(underlying, found):
        for word in wReal:
            if type(word) == list:
                word = tuple(word)
            realLex.add(word)

        for word in wFound:
            if type(word) == list:
                word = tuple(word)
            foundLex.add(word)

    actual = len(realLex)
    proposed = len(foundLex)
    matched = len(realLex.intersection(foundLex))

    # print "-- real lexicon --"

    # for xx in realLex:
    #     print cat(xx)

    # print

    # print "-- found lexicon --"

    # for xx in foundLex:
    #     print cat(xx)

    # print actual, "true words", proposed, "found words", matched, "matched"

    return precision_recall_f(matched, actual, proposed)
