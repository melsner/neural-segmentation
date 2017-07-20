from keras import backend as K
from data_handling import *
from sampling import *

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

def updateAE(ae_phon, ae_utt, embed_words, Xs, Xs_mask, segs, maxUtt, maxLen, batch_size, reverseUtt, nResample = None):
    ## Train phonological AE
    Xae_wrds, deletedChars_wrds, oneLetter_wrds = XsSeg2XaePhon(Xs,
                                                                Xs_mask,
                                                                segs,
                                                                maxLen,
                                                                nResample)

    Yae_wrds = getYae(Xae_wrds, reverseUtt, utts=False)

    print('Fitting phonological auto-encoder network')
    ae_phon.fit(Xae_wrds,
                Yae_wrds,
                batch_size=batch_size,
                epochs=1)

    ## Train utterance AE
    Xae_utts, deletedChars_utts, oneLetter_utts = XsSeg2Xae(Xs,
                                                            Xs_mask,
                                                            segs,
                                                            maxUtt,
                                                            maxLen,
                                                            nResample)
    Xae_emb = embed_words.predict(Xae_utts)
    Yae_emb = getYae(Xae_emb, reverseUtt, utts=False)

    print('Fitting utterance auto-encoder network')
    ae_utt.fit(Xae_emb,
               Yae_emb,
               batch_size=batch_size,
               epochs=1)

    return Xae_utts, deletedChars_utts, oneLetter_utts

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

def trainAEOnly(ae_full, ae_phon, ae_utt, embed_words, Xs, Xs_mask, segs, maxUtt, maxLen, trainIters, batch_size, reverseUtt, nResample = None):
    print('Training auto-encoder network')

    return updateAE(ae_phon,
                    ae_utt,
                    embed_words,
                    Xs,
                    Xs_mask,
                    segs,
                    maxUtt,
                    maxLen,
                    batch_size,
                    reverseUtt,
                    nResample)[0]

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