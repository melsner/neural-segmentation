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
        ## Set up plotting canvas
        fig.patch.set_visible(False)
        fig.suptitle('Utterance %d, Checkpoint %d' % (utt_ids[u], iteration))

        ## Plot inputs (heatmap)
        inputs = inputs_raw[u]
        inputs = inputs[np.where(1 - masks_raw[u])]
        inputs = np.swapaxes(inputs, 0, 1)
        ax_input.clear()
        ax_input.axis('off')
        ax_input.set_title('Input', loc='left')
        hm_input = ax_input.pcolor(inputs, cmap=plt.cm.Blues)

        ## Plot targets (bar chart)
        targs = targs_raw[u]
        targs = targs[np.where(1 - masks_raw[u])]
        ax_targ.clear()
        ax_targ.axis('off')
        ax_targ.set_title('Target', loc='left')
        ax_targ.set_ylim([0, 1])
        ax_targ.margins(0)
        hm_targ = ax_targ.bar(np.arange(len(targs)), targs)

        ## Plot predictions (bar chart)
        preds = preds_raw[u]
        preds = preds[np.where(1 - masks_raw[u])]
        preds = np.squeeze(preds, -1)
        ax_pred.clear()
        ax_pred.axis('off')
        ax_pred.set_title('Prediction', loc='left')
        ax_pred.set_ylim([0, 1])
        ax_pred.margins(0)
        hm_pred = ax_pred.bar(np.arange(len(preds)), preds)

        ## Save plot
        fig.savefig(logdir + '/barchart_' + prefix + '_utt' + str(utt_ids[u]) + '_iter' + str(iteration) + '.jpg')

    plt.close(fig)


def plotPredsUtt(utt_ids, model, Xae, Yae, logdir, prefix, iteration, batch_size, Xae_resamp=None, debug=False):
    ## Initialize plotting objects
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    inputs_src_raw = Xae[utt_ids]
    if not Xae_resamp is None:
        ax_input = fig.add_subplot(511)
        ax_input_resamp = fig.add_subplot(512)
        ax_targ = fig.add_subplot(513)
        ax_mean = fig.add_subplot(514)
        ax_pred = fig.add_subplot(515)
        inputs_resamp_raw = Xae_resamp[utt_ids]
        preds_raw = model.predict(inputs_resamp_raw, batch_size=batch_size)
    else:
        ax_input = fig.add_subplot(411)
        ax_targ = fig.add_subplot(412)
        ax_mean = fig.add_subplot(413)
        ax_pred = fig.add_subplot(414)
        inputs_src_raw = Xae[utt_ids]
        preds_raw = model.predict(inputs_src_raw, batch_size=batch_size)
    targs_raw = Yae[utt_ids]

    ## Plot target global mean
    mean = np.expand_dims(np.mean(Yae, axis=(0, 1, 2)), -1)
    ax_mean.axis('off')
    ax_mean.set_title('Target mean', loc='left')
    hm_mean = ax_mean.pcolor(mean, cmap=plt.cm.Blues)

    for u in range(len(utt_ids)):
        ## Set up plotting canvas
        fig.patch.set_visible(False)
        fig.suptitle('Utterance %d, Checkpoint %d' % (utt_ids[u], iteration))

        ## Plot source inputs
        inputs_src = []
        for w in range(len(inputs_src_raw[u])):
            inputs_src_w = inputs_src_raw[u, w, ...]
            inputs_src_w = inputs_src_w[np.where(inputs_src_w.any(-1))]
            inputs_src.append(inputs_src_w)
        inputs_src = np.concatenate(inputs_src)
        inputs_src = np.swapaxes(inputs_src, 0, 1)
        ax_input.clear()
        ax_input.axis('off')
        ax_input.set_title('Input (source)', loc='left')
        hm_input = ax_input.pcolor(inputs_src, cmap=plt.cm.Blues)

        ## Plot resampled inputs if applicable
        if not Xae_resamp is None:
            inputs_resamp = []
            for w in range(len(inputs_src_raw[u])):
                inputs_resamp_w = inputs_resamp_raw[u, w, ...]
                inputs_resamp_w = inputs_resamp_w[np.where(inputs_resamp_w.any(-1))]
                inputs_resamp.append(inputs_resamp_w)
            inputs_resamp = np.concatenate(inputs_resamp)
            inputs_resamp = np.swapaxes(inputs_resamp, 0, 1)
            ax_input_resamp.clear()
            ax_input_resamp.axis('off')
            ax_input_resamp.set_title('Input (resampled)', loc='left')
            hm_input = ax_input_resamp.pcolor(inputs_resamp, cmap=plt.cm.Blues)

        ## Plot reconstruction targets
        targs = []
        for w in range(len(targs_raw[u])):
            targs_w = targs_raw[u, w, ...]
            targs_w = targs_w[np.where(targs_w.any(-1))]
            targs.append(targs_w)
        targs = np.concatenate(targs)
        targs = np.swapaxes(targs, 0, 1)
        ax_targ.clear()
        ax_targ.axis('off')
        ax_targ.set_title('Target', loc='left')
        hm_targ = ax_targ.pcolor(targs, cmap=plt.cm.Blues)

        ## Plot predictions
        preds = []
        for w in range(len(preds_raw[u])):
            preds_w = preds_raw[u, w, ...]
            preds_w = preds_w[np.where(preds_w.any(-1))]
            preds.append(preds_w)
        preds = np.concatenate(preds)
        preds = np.swapaxes(preds, 0, 1)
        ax_pred.clear()
        ax_pred.axis('off')
        ax_pred.set_title('Prediction', loc='left')
        hm_pred = ax_pred.pcolor(preds, cmap=plt.cm.Blues)

        ## Save plot
        fig.savefig(logdir + '/heatmap_' + prefix + '_utt' + str(utt_ids[u]) + '_iter' + str(iteration) + '.jpg')

    plt.close(fig)

def plotPredsWrd(wrd_ids, model, Xae, Yae, logdir, prefix, iteration, batch_size, Xae_resamp=None, debug=False):
    ## Initialize plotting objects
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    inputs_src_raw = Xae[wrd_ids]
    if not Xae_resamp is None:
        ax_input_resamp = fig.add_subplot(512)
        ax_targ = fig.add_subplot(513)
        ax_mean = fig.add_subplot(514)
        ax_pred = fig.add_subplot(515)
        inputs_resamp_raw = Xae_resamp[wrd_ids]
        preds_raw = model.predict(inputs_resamp_raw, batch_size=batch_size)
    else:
        ax_input = fig.add_subplot(411)
        ax_targ = fig.add_subplot(412)
        ax_mean = fig.add_subplot(413)
        ax_pred = fig.add_subplot(414)
        preds_raw = model.predict(inputs_src_raw, batch_size=batch_size)
    targs_raw = Yae[wrd_ids]

    ## Plot target global mean
    mean = np.expand_dims(np.mean(Yae, axis=(0, 1)), -1)
    ax_mean.axis('off')
    ax_mean.set_title('Target global mean', loc='left')
    hm_mean = ax_mean.pcolor(mean, cmap=plt.cm.Blues)

    if debug:
        print('=' * 50)
        print('Segmentation details for 10 randomly-selected utterances')
    for w in range(len(wrd_ids)):
        ## Set up plotting canvas
        fig.patch.set_visible(False)
        fig.suptitle('Word %d, Checkpoint %d' % (wrd_ids[w], iteration))

        ## Plot source inputs
        inputs_src = inputs_src_raw[w, ...]
        inputs_src = inputs_src[np.where(inputs_src.any(-1))]
        inputs_src = np.swapaxes(inputs_src, 0, 1)
        ax_input.clear()
        ax_input.axis('off')
        ax_input.set_title('Input (source)', loc='left')
        hm_input = ax_input.pcolor(inputs_src, cmap=plt.cm.Blues)

        ## Plot resampled inputs if applicable
        if not Xae_resamp is None:
            inputs_resamp = inputs_resamp_raw[w, ...]
            inputs_resamp = inputs_resamp[np.where(inputs_resamp.any(-1))]
            inputs_resamp = np.swapaxes(inputs_resamp, 0, 1)
            ax_input_resamp.clear()
            ax_input_resamp.axis('off')
            ax_input_resamp.set_title('Input (resampled)', loc='left')
            hm_input = ax_input_resamp.pcolor(inputs_resamp, cmap=plt.cm.Blues)

        ## Plot reconstruction targets
        targs = targs_raw[w, ...]
        targs = targs[np.where(targs.any(-1))]
        targs = np.swapaxes(targs, 0, 1)
        ax_targ.clear()
        ax_targ.axis('off')
        ax_targ.set_title('Target', loc='left')
        hm_targ = ax_targ.pcolor(targs, cmap=plt.cm.Blues)

        ## Plot predictions
        preds = preds_raw[w, ...]
        preds = preds[np.where(preds.any(-1))]
        preds = np.swapaxes(preds, 0, 1)
        ax_pred.clear()
        ax_pred.axis('off')
        ax_pred.set_title('Prediction', loc='left')
        hm_pred = ax_pred.pcolor(preds, cmap=plt.cm.Blues)

        ## Save plot
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
            printReconstruction(utt_ids, ae_full, Xae, ctable, batchSize, reverseUtt)

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
        print('Writing solutions to file')
        writeTimeSegs(frameSegs2timeSegs(intervals, segs4evalXDoc), out_dir=logdir, TextGrid=False, dataset='cv')
        writeTimeSegs(frameSegs2timeSegs(intervals, segs4evalXDoc), out_dir=logdir, TextGrid=True, dataset='cv')
    else:
        print('Writing solutions to file')
        printSegScores(getSegScores(gold, segs4evalXDoc, acoustic), acoustic)
        writeSolutions(logdir, segs4evalXDoc[doc_list[0]], gold[doc_list[0]], batch_num, filename='seg_cv.txt')

    print()
    print('Plotting visualizations')
    if ae_net:
        if nResample:
            Xae_full, _, _ = XsSeg2Xae(Xs,
                                       Xs_mask,
                                       segs4eval,
                                       maxUtt,
                                       maxLen,
                                       nResample=None)

        plotPredsUtt(utt_ids,
                     ae_full,
                     Xae_full if nResample else Xae,
                     Yae,
                     logdir,
                     'cv',
                     batch_num,
                     batchSize,
                     Xae_resamp = Xae if nResample else None,
                     debug=debug)

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

                 debug=debug)