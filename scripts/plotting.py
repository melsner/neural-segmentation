from __future__ import print_function, division
import matplotlib.pyplot as plt
from network import *
from ae_io import *

def plotPredsSeg(utt_ids, segmenter, Xs, Xs_mask, Y, logdir, prefix, iteration, seg_shift, batch_size, debug=False):
    ## Initialize plotting objects
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax_input = fig.add_subplot(311)
    ax_targ = fig.add_subplot(312)
    ax_pred = fig.add_subplot(313)

    inputs_raw = Xs[utt_ids]
    masks_raw = Xs_mask[utt_ids]
    preds_raw = predictSegmenter(segmenter,
                                 inputs_raw,
                                 masks_raw,
                                 seg_shift,
                                 batch_size)

    targs_raw = np.expand_dims(Y[utt_ids], -1)

    for u in range(len(utt_ids)):
        inputs = inputs_raw[u]
        inputs = inputs[np.where(1 - masks_raw[u])]
        inputs = np.swapaxes(inputs, 0, 1)

        targs = targs_raw[u]
        targs = targs[np.where(1 - masks_raw[u])]

        preds = preds_raw[u]
        preds = preds[np.where(1 - masks_raw[u])]
        preds = np.squeeze(preds, -1)

        ## Create and save plots
        fig.patch.set_visible(False)
        fig.suptitle('Utterance %d, Checkpoint %d' % (utt_ids[u], iteration))

        ax_input.clear()
        ax_input.axis('off')
        ax_input.set_title('Input', loc='left')
        hm_input = ax_input.pcolor(inputs, cmap=plt.cm.Blues)

        ax_targ.clear()
        ax_targ.axis('off')
        ax_targ.set_title('Target', loc='left')
        ax_targ.set_ylim([0, 1])
        ax_targ.margins(0)
        hm_targ = ax_targ.bar(np.arange(len(targs)), targs)

        ax_pred.clear()
        ax_pred.axis('off')
        ax_pred.set_title('Prediction', loc='left')
        ax_pred.set_ylim([0, 1])
        ax_pred.margins(0)
        hm_pred = ax_pred.bar(np.arange(len(preds)), preds)

        fig.savefig(logdir + '/barchart_' + prefix + '_utt' + str(utt_ids[u]) + '_iter' + str(iteration) + '.jpg')

    plt.close(fig)


def plotPredsUtt(utt_ids, model, Xae, Yae, logdir, prefix, iteration, batch_size, debug=False):
    ## Initialize plotting objects
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax_input = fig.add_subplot(411)
    ax_targ = fig.add_subplot(412)
    ax_mean = fig.add_subplot(413)
    ax_pred = fig.add_subplot(414)
    inputs_raw = Xae[utt_ids]
    # preds_raw = model.predict(Xae[utt_ids], batch_size=batch_size)[0]
    preds_raw = model.predict(Xae[utt_ids], batch_size=batch_size)
    targs_raw = Yae[utt_ids]

    mean = np.expand_dims(np.mean(Yae, axis=(0, 1, 2)), -1)

    if debug:
        print('=' * 50)
        print('Segmentation details for 10 randomly-selected utterances')
    for u in range(len(utt_ids)):
        ## Remove word boundaries so reconstruction of entire utterance can be plotted
        inputs = []
        if debug:
            print('-' * 50)
            print('Utterance %d' % (utt_ids[u] + 1))
            sys.stdout.write('Input word lengths:')
        for w in range(len(inputs_raw[u])):
            inputs_w = inputs_raw[u, w, ...]
            inputs_w = inputs_w[np.where(inputs_w.any(-1))]
            if debug:
                sys.stdout.write(' %d' % inputs_w.shape[0])
            inputs.append(inputs_w)
        inputs = np.concatenate(inputs)
        if debug:
            print('\nInput utt length: %d' % inputs.shape[0])
            sys.stdout.write('Prediction word lengths:')
        inputs = np.swapaxes(inputs, 0, 1)
        targs = []
        for w in range(len(targs_raw[u])):
            targs_w = targs_raw[u, w, ...]
            targs_w = targs_w[np.where(targs_w.any(-1))]
            if debug:
                sys.stdout.write(' %d' % targs_w.shape[0])
            targs.append(targs_w)
        targs = np.concatenate(targs)
        if debug:
            print('\nPrediction utt length: %d' % targs.shape[0])
            sys.stdout.write('Target word lengths:')
        targs = np.swapaxes(targs, 0, 1)
        preds = []
        for w in range(len(preds_raw[u])):
            preds_w = preds_raw[u, w, ...]
            preds_w = preds_w[np.where(preds_w.any(-1))]
            if debug:
                sys.stdout.write(' %d' % preds_w.shape[0])
            preds.append(preds_w)
        preds = np.concatenate(preds)
        if debug:
            print('\nTarget utt length: %d' % preds.shape[0])
        preds = np.swapaxes(preds, 0, 1)

        ## Create and save plots
        fig.patch.set_visible(False)
        fig.suptitle('Utterance %d, Checkpoint %d' % (utt_ids[u], iteration))

        ax_input.clear()
        ax_input.axis('off')
        ax_input.set_title('Input', loc='left')
        hm_input = ax_input.pcolor(inputs, cmap=plt.cm.Blues)

        ax_targ.clear()
        ax_targ.axis('off')
        ax_targ.set_title('Target', loc='left')
        hm_targ = ax_targ.pcolor(targs, cmap=plt.cm.Blues)

        ax_mean.clear()
        ax_mean.axis('off')
        ax_mean.set_title('Target mean', loc='left')
        hm_mean = ax_mean.pcolor(mean, cmap=plt.cm.Blues)

        ax_pred.clear()
        ax_pred.axis('off')
        ax_pred.set_title('Prediction', loc='left')
        hm_pred = ax_pred.pcolor(preds, cmap=plt.cm.Blues)

        fig.savefig(logdir + '/heatmap_' + prefix + '_utt' + str(utt_ids[u]) + '_iter' + str(iteration) + '.jpg')

    plt.close(fig)

def plotPredsWrd(wrd_ids, model, Xae, Yae, logdir, prefix, iteration, batch_size, debug=False):
    ## Initialize plotting objects
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax_input = fig.add_subplot(411)
    ax_targ = fig.add_subplot(412)
    ax_mean = fig.add_subplot(413)
    ax_pred = fig.add_subplot(414)
    inputs_raw = Xae[wrd_ids]
    preds_raw = model.predict(Xae[wrd_ids], batch_size=batch_size)
    targs_raw = Yae[wrd_ids]

    mean = np.expand_dims(np.mean(Yae, axis=(0, 1)), -1)

    if debug:
        print('=' * 50)
        print('Segmentation details for 10 randomly-selected utterances')
    for w in range(len(wrd_ids)):
        ## Remove word boundaries so reconstruction of entire utterance can be plotted
        if debug:
            print('-' * 50)
            print('Word %d' % (wrd_ids[w] + 1))
            sys.stdout.write('Input word lengths:')
        inputs = inputs_raw[w, ...]
        inputs = inputs[np.where(inputs.any(-1))]
        if debug:
            sys.stdout.write(' %d' % inputs.shape[0])
        if debug:
            print('\nInput utt length: %d' % inputs.shape[0])
            sys.stdout.write('Prediction word lengths:')
        inputs = np.swapaxes(inputs, 0, 1)
        targs = targs_raw[w, ...]
        targs = targs[np.where(targs.any(-1))]
        if debug:
            sys.stdout.write(' %d' % targs.shape[0])
        if debug:
            print('\nPrediction utt length: %d' % targs.shape[0])
            sys.stdout.write('Target word lengths:')
        targs = np.swapaxes(targs, 0, 1)
        preds = preds_raw[w, ...]
        preds = preds[np.where(preds.any(-1))]
        if debug:
            sys.stdout.write(' %d' % preds.shape[0])
        if debug:
            print('\nTarget utt length: %d' % preds.shape[0])
        preds = np.swapaxes(preds, 0, 1)

        ## Create and save plots
        fig.patch.set_visible(False)
        fig.suptitle('Word %d, Checkpoint %d' % (wrd_ids[w], iteration))

        ax_input.clear()
        ax_input.axis('off')
        ax_input.set_title('Input', loc='left')
        hm_input = ax_input.pcolor(inputs, cmap=plt.cm.Blues)

        ax_targ.clear()
        ax_targ.axis('off')
        ax_targ.set_title('Target', loc='left')
        hm_targ = ax_targ.pcolor(targs, cmap=plt.cm.Blues)

        ax_mean.clear()
        ax_mean.axis('off')
        ax_mean.set_title('Target mean', loc='left')
        hm_mean = ax_mean.pcolor(mean, cmap=plt.cm.Blues)

        ax_pred.clear()
        ax_pred.axis('off')
        ax_pred.set_title('Prediction', loc='left')
        hm_pred = ax_pred.pcolor(preds, cmap=plt.cm.Blues)

        fig.savefig(logdir + '/heatmap_' + prefix + '_wrd' + str(wrd_ids[w]) + '_iter' + str(iteration) + '.jpg')

    plt.close(fig)

def evalCrossVal(Xs, Xs_mask, gold, doc_list, doc_indices, utt_ids, otherParams, maxLen, maxUtt, raw_total, logdir,
                 segShift, batchSize, reverseUtt, iteration, batch_num, acoustic=False, nResample=None, ae_full=None,
                 segmenter=None, debug=False):
    if acoustic:
        intervals, GOLDWRD, GOLDPHN, vad = otherParams
    else:
        ctable = otherParams
    ae_net = ae_full != None
    seg_net = segmenter != None
    print()
    print('Performing system evaluation')
    print('Segmenting cross-validation set')
    if seg_net:
        preds = predictSegmenter(segmenter,
                                 Xs,
                                 Xs_mask,
                                 segShift,
                                 batchSize)
        segs4eval = (preds > 0.5).astype(np.int8)
        if acoustic:
            segs4eval[np.where(vad)] = 1.
        else:
            segs4eval[:, 0, ...] = 1.
        segs4eval[np.where(Xs_mask)] = 0.
        cvSeg = segs4eval.sum()

    if ae_net:
        Xae, deletedChars, oneLetter = XsSeg2Xae(Xs,
                                                 Xs_mask,
                                                 segs4eval,
                                                 maxUtt,
                                                 maxLen,
                                                 nResample)
        Yae = getYae(Xae, reverseUtt)
        print('Computing network losses on cross-validation set')
        cvAELoss = lossXChar(ae_full, Xae, Yae, batchSize, acoustic, 'mse' if acoustic else 'xent')
        cvDel = deletedChars.sum()
        cvOneL = oneLetter.sum()

        if not acoustic:
            print('')
            print('Example reconstruction of learned segmentation')
            printReconstruction(10, ae_full, Xae, ctable, batchSize, reverseUtt)

    segs4evalXDoc = dict.fromkeys(doc_list)
    for doc in segs4evalXDoc:
        s, e = doc_indices[doc]
        segs4evalXDoc[doc] = segs4eval[s:e]
        if acoustic:
            masked_proposal = np.ma.array(segs4evalXDoc[doc], mask=Xs_mask[s:e])
            segs4evalXDoc[doc] = masked_proposal.compressed()

    print('Scoring segmentation of cross-validation set')
    segScore = writeLog(batch_num,
                        iteration,
                        cvAELoss if ae_net else None,
                        None,
                        None,
                        cvDel if ae_net else None,
                        cvOneL if ae_net else None,
                        cvSeg,
                        gold,
                        segs4evalXDoc,
                        logdir,
                        intervals = intervals if acoustic else None,
                        acoustic = acoustic,
                        print_headers = not os.path.isfile(logdir + '/log_cv.txt'),
                        filename = 'log_cv.txt')

    print('Total frames:', raw_total)
    if ae_net:
        print('Auto-encoder loss:', cvAELoss)
        print('Deletions:', cvDel)
        print('One letter words:', cvOneL)
    print('Total segmentation points:', cvSeg)

    if acoustic:
        if GOLDWRD:
            print('Word segmentation scores:')
            printSegScores(segScore['wrd'], acoustic)
        if GOLDPHN:
            print('Phone segmentation scores:')
            printSegScores(segScore['phn'], acoustic)
        writeTimeSegs(frameSegs2timeSegs(intervals, segs4evalXDoc), out_dir=logdir, TextGrid=False, dataset='cv')
        writeTimeSegs(frameSegs2timeSegs(intervals, segs4evalXDoc), out_dir=logdir, TextGrid=True, dataset='cv')
    else:
        printSegScores(getSegScores(gold, segs4evalXDoc, acoustic), acoustic)
        writeSolutions(logdir, segs4evalXDoc[doc_list[0]], gold[doc_list[0]], batch_num, filename='seg_cv.txt')

    print()
    print('Plotting visualizations')
    if ae_net:
        plotPredsUtt(utt_ids,
                     ae_full,
                     Xae,
                     Yae,
                     logdir,
                     'cv',
                     batch_num,
                     batchSize,
                     debug)

    plotPredsSeg(utt_ids,
                 segmenter,
                 Xs,
                 Xs_mask,
                 segs4eval,
                 logdir,
                 'cv',
                 batch_num,
                 segShift,
                 batchSize,
                 debug)