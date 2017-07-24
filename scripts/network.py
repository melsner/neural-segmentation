from keras import backend as K
from keras.engine.training import _make_batches
from keras.utils.generic_utils import Progbar
from keras.engine.training import _slice_arrays
from data_handling import *
from sampling import *

## Define a Keras backend function with batching
def makeFunction(f_in):
    def f_out(input, batch_size=32, learning_phase=0, verbose=0):
        return predictLoop(f_in, input, batch_size, learning_phase, verbose)
    return f_out

## Static method version of keras.engine.training.Model._predict_loop to allow
## batching in Keras backend functions
def predictLoop(f, ins, batch_size=32, learning_phase=0, verbose=0):
    """Abstract method to loop over some data in batches.
    # Arguments
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size.
        verbose: verbosity mode.
    # Returns
        Array of predictions (if the model has a single output)
        or list of arrays of predictions
        (if the model has multiple outputs).
    """
    if type(ins) is not list:
        ins = [ins]
    if ins and hasattr(ins[0], 'shape'):
        samples = ins[0].shape[0]
    else:
        # May happen if we are running `predict` without Numpy input data,
        # i.e. if all inputs to the models are data tensors
        # instead of placeholders.
        # In that case we will run `predict` over a single batch.
        samples = batch_size
        verbose = 2
    outs = []
    if verbose == 1:
        progbar = Progbar(target=samples)
    batches = _make_batches(samples, batch_size)
    index_array = np.arange(samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        if ins and isinstance(ins[-1], float):
            # Do not slice the training phase flag.
            ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
        else:
            ins_batch = _slice_arrays(ins, batch_ids)

        batch_outs = f(ins_batch + [learning_phase])
        if not isinstance(batch_outs, list):
            batch_outs = [batch_outs]
        if batch_index == 0:
            for batch_out in batch_outs:
                shape = (samples,) + batch_out.shape[1:]
                outs.append(np.zeros(shape, dtype=batch_out.dtype))

        for i, batch_out in enumerate(batch_outs):
            outs[i][batch_start:batch_end] = batch_out
        if verbose == 1:
            progbar.update(batch_end)
    if len(outs) == 1:
        return outs[0]
    return outs

def mask_output(x, input, mask_value, reverseUtt):
    if reverseUtt:
        m = K.any(K.not_equal(K.reverse(input, (1,2)), mask_value), axis=-1, keepdims=True)
    else:
        m = K.any(K.not_equal(input, mask_value), axis=-1, keepdims=True)
    x *= K.cast(m, 'float32')
    x += 1 - K.cast(m, 'float32')
    return x


def masked_categorical_crossentropy(y_true, y_pred):
    mask = K.cast(K.expand_dims(K.any(y_true, -1), axis=-1), 'float32')
    y_pred *= mask
    y_pred += 1-mask
    y_pred += 1-mask
    losses = K.categorical_crossentropy(y_pred, y_true)
    losses *= K.squeeze(mask, -1)
    ## Normalize by number of real segments, using a small non-zero denominator in cases of padding characters
    ## in order to avoid division by zero
    #losses /= (K.mean(mask) + (1e-10*(1-K.mean(mask))))
    return losses

def masked_mean_squared_error(y_true, y_pred):
    y_pred = y_pred * K.cast(K.any(K.reverse(y_true, (1,2)), axis=-1, keepdims=True), 'float32')
    return K.mean(K.square(y_pred - y_true), axis=-1)

def masked_categorical_accuracy(y_true, y_pred):
    mask = K.cast(K.expand_dims(K.greater(K.argmax(y_true, axis=-1), 0), axis=-1), 'float32')
    accuracy = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'float32')
    accuracy *= K.squeeze(mask, -1)
    ## Normalize by number of real segments, using a small non-zero denominator in cases of padding characters
    ## in order to avoid division by zero
    #accuracy /= (K.mean(mask) + (1e-10*(1-K.mean(mask))))
    return accuracy

def wrdEmbdTargets(wrd_emb, Xae, batch_size, reverseUtt):
    if reverseUtt:
        get_embds = lambda: np.flip(wrd_emb.predict(Xae, batch_size=batch_size), 1)
    else:
        get_embds = lambda: wrd_emb.predict(Xae, batch_size=batch_size)
    return get_embds()

def updateAE(ae_full, ae_phon, ae_utt, embed_words, Xs, Xs_mask, segs, maxUtt, maxLen, batch_size, reverseUtt,
             nResample=None, fitParts=True, fitFull=False, embed_word=None, embed_words_reconst=None):
    Xae_full, deletedChars_full, oneLetter_full = XsSeg2Xae(Xs,
                                                            Xs_mask,
                                                            segs,
                                                            maxUtt,
                                                            maxLen,
                                                            nResample)
    Yae_full = getYae(Xae_full, reverseUtt)

    if fitParts:
        ## Train phonological AE
        Xae_wrds, deletedChars_wrds, oneLetter_wrds = XsSeg2XaePhon(Xs,
                                                                    Xs_mask,
                                                                    segs,
                                                                    maxLen,
                                                                    nResample)

        Yae_wrds = getYae(Xae_wrds, reverseUtt)
        print('Fitting phonological auto-encoder network')
        ae_phon.fit(Xae_wrds,
                    Yae_wrds,
                    batch_size=batch_size,
                    epochs=1)

        ## Train utterance AE
        wordMask = Yae_full.any(axis=(-2,-1))

        ## Extract input word embeddings
        Xae_utt = embed_words(Xae_full, learning_phase=1, batch_size=batch_size)
        # Xae_utt = embed_words([Xae_full, 1])

        ## Extract target word embeddings
        Yae_utt = embed_words(Xae_full, learning_phase=0, batch_size=batch_size)
        # Yae_utt = embed_words([Xae_full, 0])
        Yae_utt = getYae(Yae_utt, reverseUtt)

        # Xae_emb2 = embed_word(Xae_full[0], learning_phase=0, batch_size=batch_size)
        # Xae_emb3 = embed_words_reconst(Xae_full, learning_phase=0, batch_size=batch_size)
        # Xae_emb4 = ae_utt.predict([np.expand_dims(Xae_utt[0],0),np.expand_dims(wordMask[0],0)])
        # print('Input embeddings')
        # print(np.squeeze(Xae_utt[0,...,0]))
        # print('Input embeddings (generated by sindle word encoder)')
        # print(Xae_emb2[...,0])
        # print('Reconstructed embeddings from full system')
        # print(np.squeeze(Xae_emb3[0,...,0]))
        # print('Reconstructed embeddings from utterance AE')
        # print(np.squeeze(Xae_emb4[0,...,0]))
        # print('Target embeddings')
        # print(np.squeeze(Yae_utt[0,...,0]))
        # raw_input()

        print('Fitting utterance auto-encoder network')
        ae_utt.fit([Xae_utt,wordMask],
                   Yae_utt,
                   batch_size=batch_size,
                   epochs=1)

    if fitFull:
        print('Fitting full auto-encoder network')
        ae_full.fit(Xae_full,
                    Yae_full,
                    batch_size,
                    epochs=1)

    eval = ae_full.evaluate(Xae_full, Yae_full, batch_size, verbose=0)
    print('Full AE network loss: %.4f' % eval[0])
    if len(eval) > 1:
        print('Full AE network accuracy: %.4f' % eval[1])

    return Xae_full, deletedChars_full, oneLetter_full, eval

def updateSegmenter(segmenter, Xs, Xs_mask, segs, seg_shift, batch_size):
    charDim = Xs.shape[-1]
    maxChar = Xs_mask.shape[-1]
    seg_inputs = np.zeros((len(Xs), maxChar + seg_shift, charDim))
    seg_inputs[:, :maxChar, :] = Xs
    seg_mask = np.zeros((len(Xs_mask), maxChar + seg_shift))
    seg_mask[:, seg_shift:] = Xs_mask
    seg_mask = np.expand_dims(seg_mask, -1)
    seg_targets = np.zeros((len(segs), maxChar + seg_shift, 1))
    seg_targets[:, seg_shift:, :] = segs
    segHist = segmenter.fit([seg_inputs, seg_mask],
                            seg_targets,
                            batch_size=batch_size,
                            epochs=1)
    return segHist

def predictSegmenter(segmenter, Xs, Xs_mask, seg_shift, batch_size):
    charDim = Xs.shape[-1]
    maxChar = Xs_mask.shape[-1]
    seg_inputs = np.zeros((len(Xs), maxChar + seg_shift, charDim))
    seg_inputs[:, :maxChar, :] = Xs
    seg_mask = np.zeros((len(Xs_mask), maxChar + seg_shift))
    seg_mask[:, seg_shift:] = Xs_mask
    seg_mask = np.expand_dims(seg_mask, -1)
    return segmenter.predict([seg_inputs,seg_mask], batch_size=batch_size)[:, seg_shift:, :]

def trainAEOnly(ae_full, ae_phon, ae_utt, embed_words, Xs, Xs_mask, segs, maxUtt, maxLen, trainIters, batch_size,
                reverseUtt, nResample=None, fitParts=True, fitFull=False):
    print('Training auto-encoder network')

    return updateAE(ae_full,
                    ae_phon,
                    ae_utt,
                    embed_words,
                    Xs,
                    Xs_mask,
                    segs,
                    maxUtt,
                    maxLen,
                    batch_size,
                    reverseUtt,
                    nResample,
                    fitParts,
                    fitFull)

def trainAEPhonOnly(ae, Xs, Xs_mask, segs, maxLen, trainIters, batch_size, reverseUtt, nResample = None):
    print('Training phonological auto-encoder network')

    ## Preprocess input data
    Xae, deletedChars, oneLetter = XsSeg2XaePhon(Xs,
                                                 Xs_mask,
                                                 segs,
                                                 maxLen,
                                                 nResample)

    ## Randomly permute samples
    p, p_inv = getRandomPermutation(len(Xae))
    Xae = Xae[p]
    if reverseUtt:
        Yae = np.flip(Xae, 1)
    else:
        Yae = Xae

    ae.fit(Xae,
           Yae,
           batch_size=batch_size,
           epochs=trainIters)

    return Xae[p_inv]



def trainSegmenterOnly(segmenter, Xs, Xs_mask, Y, trainIters, batch_size, seg_shift):
    print('Training segmenter network')

    for i in range(trainIters):
        updateSegmenter(segmenter,
                        Xs,
                        Xs_mask,
                        Y,
                        seg_shift,
                        batch_size)

    print('Getting model predictions for evaluation')
    preds = predictSegmenter(segmenter,
                             Xs,
                             Xs_mask,
                             seg_shift,
                             batch_size)

    segsProposal = (preds > 0.5) * np.expand_dims(1-Xs_mask, -1)

    return segsProposal