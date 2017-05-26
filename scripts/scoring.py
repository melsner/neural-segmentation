from __future__ import print_function, division
import sys, numpy as np, time
from data_handling import charSeq2WrdSeq


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
                print("Warning: surface string mismatch:", cat1, cat2)
                warned = 1
            elif warned == 1:
                print("Warning: more mismatches")
                warned += 1
            #sys.exit(1)

        trueBreaks = set(getBreaks(wReal))

        foundBreaks = set(getBreaks(wFound))

        actual += len(trueBreaks)
        proposed += len(foundBreaks)
        matched += len(trueBreaks.intersection(foundBreaks))

    return precision_recall_f(matched, actual, proposed)

def scoreFrameSegs(actual, found, tol=0.03, verbose=False):
    matched_b = matched_w = 0
    matched_b_start = matched_b_end = 0
    # Initialize to 1 because of final fencepost
    proposed_b = actual_b = 1
    proposed_w = actual_w = 1

    # Initialize pointers that will be use to crawl the segmentations
    a_ptr = f_ptr = 0

    # Whether or not the last segmentation check was a match.
    # Will be used to determine word matches.
    last_match = 0

    # Variables that contain the most recent segment endpoint
    # Will be used for checking for skipped silence.
    last_a = actual[a_ptr][0]
    a_e = last_a
    last_f = found[f_ptr][0]
    f_e = last_f
    same_start = np.allclose(last_a, last_f, rtol= 0., atol = tol)
    a_jump_count = f_jump_count = joint_jump_count = int(same_start)
    last_matched_end = 0.

    # As long as neither list is consumed...
    while a_ptr < len(actual) and f_ptr < len(found):
        
        # Next actual segment
        a_s, a_e = actual[a_ptr]
        # Next found segment
        f_s, f_e = found[f_ptr]

       # print((a_s,a_e),(f_s,f_e))
        # Check whether any silence was skipped since the last segmentation.
        # If true, additional fenceposts and checks must be added
        # for the end of the last segmentation, since only the start
        # was checked.
        a_jumped = not np.allclose(a_s, last_a)
        f_jumped = not np.allclose(f_s, last_f)

        if a_jumped or f_jumped:
            if verbose:
                if a_jumped:
                    print('A will jump.')
                if f_jumped:
                    print('F will jump.')
            # Add a match if the interval ends matched for the previous segmentation.
            if np.allclose(actual[a_ptr-a_jumped][a_jumped], found[f_ptr-f_jumped][f_jumped], rtol=0., atol=tol):
                if not np.allclose(last_matched_end, actual[a_ptr-a_jumped][a_jumped], rtol=0., atol=tol):
                    if verbose:
                        print('Ends match before skipped silence.')
                    matched_b_end += 1
                    if verbose and last_match:
                        print('Word match.')
                    matched_w += last_match
                    # Current segmentation can't finish a word since there was preceding
                    # silence in one or both segmentations.
                    last_match = 0
                    last_matched_end = actual[a_ptr-a_jumped][a_jumped]

        if verbose:
            print('')
            print('Last gold: %s. Last found: %s.' %(last_a,last_f))
            print('Gold: %s %s. Found: %s %s.' %(a_s,a_e,f_s,f_e))

        # Check whether current start points align within tolerance window 
        if np.allclose(a_s, f_s, rtol = 0., atol = tol):
            if verbose:
                print('Starts match.')
            matched_b_start += 1
            # It's a word match if both this check and the previous one were matches
            matched_w += last_match
            if verbose and last_match:
                print('Word match (see preceding comparison).')
            last_match = 1
            # Pop off the segmentation point from both lists
            a_ptr += 1
            actual_b += 1
            actual_w += 1
            f_ptr += 1
            proposed_b += 1
            proposed_w += 1
            last_a = a_e
            if a_jumped:
                # Add an extra actual if there was skipped silence in actual
                a_jump_count += 1
                actual_b += 1
            last_f = f_e
            if f_jumped:
                # Add an extra proposed if there was skipped silence in proposed
                f_jump_count += 1
                proposed_b += 1
            if a_jumped and f_jumped:
                joint_jump_count += 1
        elif f_s < a_s:
            # Mismatch and found segment starts earlier than actual
            # Pop off found segment
            f_ptr += 1
            proposed_b += 1
            proposed_w += 1
            last_f = f_e
            if f_jumped:
                f_jump_count += 1
                proposed_b += 1
            # This was not a match
            last_match = 0
        else: # a_s < f_s
            # Same...
            a_ptr += 1
            actual_b += 1
            actual_w += 1
            last_a = a_e
            if a_jumped:
                a_jump_count += 1
                actual_b += 1
            last_match = 0

    final_match = False

    # Check the final bound for a match
    while a_ptr < len(actual) and not final_match:
        a_e = actual[a_ptr][1]
        if np.allclose(a_e, f_e, rtol = 0., atol = tol):
            matched_b_end += 1
            matched_w += last_match
            final_match = True
        a_ptr += 1
        actual_b += 1
        actual_w += 1

    while f_ptr < len(found) and not final_match:
        f_e = found[f_ptr][1]
        if np.allclose(a_e, f_e, rtol = 0., atol = tol):
            matched_b_end += 1
            matched_w += last_match
            final_match = True
        f_ptr += 1
        proposed_b += 1
        proposed_w += 1

    if not final_match and np.allclose(a_e, f_e, rtol = 0., atol = tol):
        matched_b_end += 1
        matched_w += last_match

 
    matched_b = matched_b_start+matched_b_end

    if verbose:
        print('Number of shared segment starts = %s' %matched_b_start)
        print('Number of shared segment ends at speech interval ends = %s' %matched_b_end)
        print('Number of shared speech intervals = %s' %joint_jump_count)
    
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

def getSegScore(segsGold, segsProposal):
    return list(scoreFrameSegs(segsGold, segsProposal))

def getSegScores(segsGold, segsProposal, acoustic=False, out_file=None):
    scores = dict.fromkeys(segsProposal.keys())
    if not out_file:
        out_file = sys.stdout
    bm_tot = ba_tot = bP_tot = swm_tot = swa_tot = swP_tot = 0
    for doc in sorted(segsProposal.keys()):
        if doc in segsGold:
            if acoustic:
                scores[doc] = getSegScore(segsGold[doc], segsProposal[doc])
                (bm,ba,bP), (bp,br,bf), (swm,swa,swP), (swp,swr,swf) = scores[doc] 
                bm_tot, ba_tot, bP_tot = bm_tot+bm, ba_tot+ba, bP_tot+bP
                swm_tot, swa_tot, swP_tot = swm_tot+swm, swa_tot+swa, swP_tot+swP
            else:
                segmented = charSeq2WrdSeq(segsProposal[doc], segsGold[doc])
                (bp,br,bf) = scoreBreaks(segsGold[doc], segmented)
                (swp,swr,swf) = scoreWords(segsGold[doc], segmented)
                (lp,lr,lf) = scoreLexicon(segsGold[doc], segmented)
                scores[doc] = (bp,br,bf), (swp,swr,swf), (lp,lr,lf)
        else:
            print('Warning: Document ID "%s" in training data but not in gold. Skipping evaluation for this file.' %doc,file=out_file)
    if acoustic:
        bp,br,bf = precision_recall_f(bm_tot,ba_tot,bP_tot)
        swp,swr,swf = precision_recall_f(swm_tot,swa_tot,swP_tot)
        scores['##overall##'] = (bm_tot,ba_tot,bP_tot), (bp,br,bf), (swm_tot,swa_tot,swP_tot), (swp,swr,swf)
    return scores

def printSegScore(score, doc, acoustic=False, out_file=None):
    if not out_file:
        out_file = sys.stdout
    if acoustic:
        _, (bp,br,bf), _, (swp,swr,swf) = score
        print('Score for document "%s":' %doc, file=out_file)
        print("BP %4.2f BR %4.2f BF %4.2f" % (100 * bp, 100 * br, 100 * bf), file=out_file)
        print("SP %4.2f SR %4.2f SF %4.2f" % (100 * swp, 100 * swr, 100 * swf), file=out_file)
    else:
        (bp,br,bf), (swp,swr,swf), (lp,lr,lf) = score
        print('Score for document "%s":' %doc, file=out_file)
        print("BP %4.2f BR %4.2f BF %4.2f" % (100 * bp, 100 * br, 100 * bf), file=out_file)
        print("SP %4.2f SR %4.2f SF %4.2f" % (100 * swp, 100 * swr, 100 * swf), file=out_file)
        print("LP %4.2f LR %4.2f LF %4.2f" % (100 * lp, 100 * lr, 100 * lf), file=out_file)

def printSegScores(scores, acoustic=False, out_file=None):
    if not out_file:
        out_file = sys.stdout
    print('Per-document scores:', file=out_file)
    for doc in sorted(scores.keys()):
        if scores[doc] != None and doc != '##overall##':
            printSegScore(scores[doc], doc, acoustic, out_file)
    if acoustic:
        printSegScore(scores['##overall##'], 'All Data', acoustic, out_file)


