from __future__ import division
import sys, numpy as np

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

def scoreFrameSegs(actual, found, tol=0.02):
    matched_b = matched_w = 0
    # Use word counts
    proposed_w = len(found)
    actual_w = len(actual)

    # Will initially be word counts + 1 for final fencepost.
    # Will need to increment for any skipped regions
    proposed_b, actual_b = proposed_w+1,actual_w+1

    # Initialize pointers that will be use to crawl the segmentations
    a_ptr = f_ptr = 0

    # Whether or not the last segmentation check was a match.
    # Will be used to determine word matches.
    last_match = 0

    # Variables that contain the most recent segment endpoint
    # Will be used for checking for skipped silence.
    last_a = actual[a_ptr][0]
    last_f = found[f_ptr][0]
    
    # As long as neither list is consumed...
    while a_ptr < actual_w and f_ptr < proposed_w:
        
        # Next actual segment
        a_s, a_e = actual[a_ptr]
        # Next found segment
        f_s, f_e = found[f_ptr]

        # Check whether any silence was skipped since the last segmentation.
        # If true, additional fenceposts and checks must be added
        # for the end of the last segmentation, since only the start
        # was checked.
        a_jumped = not np.allclose(a_s, last_a)
        f_jumped = not np.allclose(f_s, last_f)
        if a_jumped or f_jumped:
            # Add a match if the interval ends matched for the previous segmentation.
            if np.allclose(actual[a_ptr-a_jumped][1], found[f_ptr-f_jumped][1], rtol= 0., atol = tol):
                matched_b += 1
                matched_w += last_match
                # Current segmentation can't finish a word since there was preceding
                # silence in one or both segmentations.
                last_match = 0
            if a_jumped:
                # Add an extra actual if there was skipped silence in actual
                actual_b += 1
            if f_jumped:
                # Add an extra proposed if there was skipped silence in proposed
                proposed_b += 1

        # Check whether current start points align within tolerance window 
        if np.allclose(a_s, f_s, rtol = 0., atol = tol):
            matched_b += 1
            # It's a word match if both this check and the previous one were matches
            matched_w += last_match
            last_match = 1
            # Pop off the segmentation point from both lists
            a_ptr += 1
            f_ptr += 1
        elif f_s < a_s:
            # Mismatch and found segment starts earlier than actual
            # Pop off found segment
            f_ptr += 1
            # This was not a match
            last_match = 0
        else: # a_s < f_s
            # Same...
            a_ptr += 1
            last_match = 0
        # Update record of most recent segmentation endpoint
        # for silence checks at next iter
        last_a = a_e
        last_f = f_e

    # The the final bound for a match
    if np.allclose(a_e, f_e, rtol = 0., atol = 0.02):
            matched_b += 1
            matched_w += last_match
  
    return (matched_b, actual_b, proposed_b), \
           precision_recall_f(matched_b, actual_b, proposed_b), \
           (matched_w, actual_w, proposed_w), \
           precision_recall_f(matched_w, actual_w, proposed_w)

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

def countIntersectWithTol(s1, s2, tol=0.02):
    s1 = set(s1)
    s2 = set(s2)
    intersection_size = 0
    s2 = s2.copy()
    for x in s1:
        for y in s2:
            if np.allclose(x[0], y[0], rtol = 0., atol = 0.02) \
                        and np.allclose(x[1], y[1], rtol = 0., atol = 0.02):
                intersection_size +=1
                s2.remove(y)
                break
    return intersection_size

def scoreFrameWords(underlying, found):
    actual = len(underlying)
    proposed = len(found)
    matched = countIntersectWithTol(underlying, found)

    return precision_recall_f(matched,actual,proposed)

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
