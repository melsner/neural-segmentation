from __future__ import print_function, division
from keras import backend as K
from keras.engine.training import _make_batches
from keras.utils.generic_utils import Progbar
from keras.engine.training import _slice_arrays
from keras.models import Model, Sequential, load_model
from keras import optimizers, metrics
from keras.layers import *
from keras import backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
    import plotly.offline

    # def authenticatePlotly():
    #     with open('plot.ly.config', 'rb') as f:
    #         config = f.readlines()
    #     username = config[0].strip()
    #     api_key = config[1].strip()
    #     plotly.tools.set_credentials_file(username=username, api_key=api_key)
    #
    # try:
    #     authenticatePlotly()
    #     usePlotly = True
    # except:
    #     usePlotly = False
    usePlotly = True
except:
    usePlotly = False
import pickle
import os
import h5py
from data_handling import *
from scoring import getSegScores, printSegScores
from ae_io import writeLog, writeCharSegs, writeTimeSegs





##################################################################################
##################################################################################
##
##  Keras Utility Functions
##
##################################################################################
##################################################################################

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

def sparse_xent(y_true,y_pred):
    out = metrics.sparse_categorical_crossentropy(y_true, y_pred)
    ## This is necessary because of a bug in keras' sparse cross entropy with hierarchical time series
    out = K.reshape(out, K.shape(y_true)[:-1])
    return out

def sparse_acc(y_true,y_pred):
    return metrics.sparse_categorical_accuracy(y_true, y_pred)

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

def masked_categorical_accuracy(y_true, y_pred):
    mask = K.cast(K.expand_dims(K.greater(K.argmax(y_true, axis=-1), 0), axis=-1), 'float32')
    accuracy = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'float32')
    accuracy *= K.squeeze(mask, -1)
    ## Normalize by number of real segments, using a small non-zero denominator in cases of padding characters
    ## in order to avoid division by zero
    #accuracy /= (K.mean(mask) + (1e-10*(1-K.mean(mask))))
    return accuracy

def loss_array(y_true, y_pred, fn):
    return fn(y_true, y_pred)

## Hacked from Keras core code
def save_optimizer_weights(model, path):
    with h5py.File(path, 'w') as f:
        symbolic_weights = getattr(model.optimizer, 'weights')
        if symbolic_weights:
            optimizer_weights_group = f.create_group('optimizer_weights')
            weight_values = K.batch_get_value(symbolic_weights)
            weight_names = []
            for i, (w, val) in enumerate(zip(symbolic_weights,
                                             weight_values)):
                # Default values of symbolic_weights is /variable
                # for theano and cntk
                if K.backend() == 'theano' or K.backend() == 'cntk':
                    if hasattr(w, 'name'):
                        if w.name.split('/')[-1] == 'variable':
                            name = str(w.name) + '_' + str(i)
                        else:
                            name = str(w.name)
                    else:
                        name = 'param_' + str(i)
                else:
                    if hasattr(w, 'name') and w.name:
                        name = str(w.name)
                    else:
                        name = 'param_' + str(i)
                weight_names.append(name.encode('utf8'))
            optimizer_weights_group.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                param_dset = optimizer_weights_group.create_dataset(
                    name,
                    val.shape,
                    dtype=val.dtype)
                if not val.shape:
                    # scalar
                    param_dset[()] = val
                else:
                    param_dset[:] = val
        f.flush()

## Hacked from Keras core code
def load_optimizer_weights(model, path):
    with h5py.File(path, 'r') as f:
        if 'optimizer_weights' in f:
            optimizer_weights_group = f['optimizer_weights']
            optimizer_weight_names = [n.decode('utf8') for n in
                                      optimizer_weights_group.attrs['weight_names']]
            optimizer_weight_values = [optimizer_weights_group[n] for n in
                                       optimizer_weight_names]
            try:
                model.optimizer.set_weights(optimizer_weight_values)
            except ValueError:
                warnings.warn('Error in loading the saved optimizer '
                              'state. As a result, your model is '
                              'starting with a freshly initialized '
                              'optimizer.')




##################################################################################
##################################################################################
##
##  Network Classes
##
##################################################################################
##################################################################################

class AE(object):
    def __init__(self, ae_full, ae_utt=None, ae_phon=None, embed_word=None, embed_words=None, embed_words_reconst=None,
                 word_decoder=None, words_decoder=None, loss_tensor=None, latentDim=None, charDropout=None,
                 wordDropout=None, fitType=None):
        self.full = ae_full
        self.utt = ae_utt
        self.phon = ae_phon
        self.embed_word = embed_word
        self.embed_words = embed_words
        self.embed_words_reconst = embed_words_reconst
        self.word_decoder = word_decoder
        self.words_decoder = words_decoder
        self.loss_tensor = loss_tensor
        self.latentDim = latentDim
        self.charDropout = charDropout
        self.wordDropout = wordDropout
        self.fitType = fitType
        self.fitFull = fitType in ['full', 'both']
        self.fitParts = fitType in ['parts', 'both']

    def decode_word(self, input, batch_size=128, verbose=0):
        return self.word_decoder.predict(input, batch_size=batch_size, verbose=verbose)

    def decode_words(self, input, batch_size=128, verbose=0):
        return self.words_decoder.predict(input, batch_size=batch_size, verbose=verbose)

    ## Faster but more memory intensive
    def predictA(self, input, batch_size=128, verbose=0):
        return self.full.predict(input, batch_size=batch_size, verbose=verbose)

    ## Slower but more memory efficient
    def predictB(self, input, batch_size=128, verbose=0):
        output = self.embed_words(input, learning_phase=0, batch_size=batch_size, verbose=verbose)
        output = self.utt.predict(output, batch_size=batch_size, verbose=verbose)
        output = self.decode_words([output, input], batch_size=batch_size, verbose=verbose)
        return output

    def predict(self, input, batch_size=128, verbose=0):
        if self.fitType == 'full':
            ## fitType 'full' requires version A because the utterance model doesn't auto-encode
            return self.predictA(input, batch_size, verbose=verbose)
        return self.predictB(input, batch_size, verbose=verbose)

    ## Faster but more memory intensive
    def evaluateA(self, input, target, batch_size=128, verbose=0):
        if verbose != 0:
            print('Encoding/decoding full input')
        eval = self.full.evaluate(input, target, batch_size=batch_size, verbose=verbose)
        if type(eval) is not list:
            eval = [eval]
        return eval

    ## Slower but more memory efficient
    def evaluateB(self, input, target, batch_size=128, verbose=0):
        if verbose != 0:
            print('Encoding word sequences')
        output = self.embed_words(input, learning_phase=0, batch_size=batch_size, verbose=verbose)
        if verbose != 0:
            print('Encoding/decoding utterances')
        output = self.utt.predict(output, batch_size=batch_size, verbose=verbose)
        if verbose != 0:
            print('Decoding word sequences')
        output = self.words_decoder.evaluate([output, input], target, batch_size=batch_size, verbose=verbose)
        if type(output) is not list:
            output = [output]
        return output

    def evaluate(self, input, target, batch_size=128, verbose=0):
        if self.fitType == 'full':
            ## fitType 'full' requires version A because the utterance model doesn't auto-encode
            return self.evaluateA(input, target, batch_size, verbose=verbose)
        return self.evaluateB(input, target, batch_size, verbose=verbose)

    def update(self, Xs, Xs_mask, segs, maxUtt, maxLen, reverseUtt=False, batch_size=128,
                 nResample=None, nEpoch=1, fitParts=True, fitFull=True, verbose=True):
        if verbose:
            print('Segmenting input sequence for full AE')
        Xae_full, deletedChars_full, oneLetter_full = XsSeg2Xae(Xs,
                                                                Xs_mask,
                                                                segs,
                                                                maxUtt,
                                                                maxLen,
                                                                nResample)
        Yae_full = getYae(Xae_full, reverseUtt)

        if fitParts:
            ## Train phonological AE
            if verbose:
                print('Segmenting input sequence for phon AE')
            Xae_wrds, deletedChars_wrds, oneLetter_wrds = XsSeg2XaePhon(Xs,
                                                                        Xs_mask,
                                                                        segs,
                                                                        maxLen,
                                                                        nResample)

            Yae_wrds = getYae(Xae_wrds, reverseUtt)

            if verbose:
                print('Fitting phonological auto-encoder network')
            self.phon.fit(Xae_wrds,
                          Yae_wrds,
                          shuffle=True,
                          batch_size=batch_size,
                          epochs=nEpoch,
                          verbose=int(verbose))

            ## Extract input word embeddings
            if verbose:
                print('Getting input word embeddings (dropout on)')
            Xae_utt = self.embed_words(Xae_full, learning_phase=1, batch_size=batch_size, verbose=int(verbose))

            ## Extract target word embeddings
            if self.charDropout != 0 or self.wordDropout != 0:
                if verbose:
                    print('Getting target word embeddings (dropout off)')
                Yae_utt = self.embed_words(Xae_full, learning_phase=0, batch_size=batch_size, verbose=int(verbose))
            else:
                Yae_utt = Xae_utt
            Yae_utt = getYae(Yae_utt, reverseUtt)

            # ## Debugging printouts
            # Xae_emb2 = self.embed_word(Xae_full[0], learning_phase=0, batch_size=batch_size)
            # Xae_emb3 = self.embed_words_reconst(Xae_full, learning_phase=0, batch_size=batch_size)
            # Xae_emb4 = self.utt.predict(np.expand_dims(Xae_utt[0],0))
            # print('Input embeddings')
            # print(np.squeeze(Xae_utt[0,...,0]))
            # print('Input embeddings (generated by single word encoder)')
            # print(Xae_emb2[...,0])
            # print('Reconstructed embeddings from full system')
            # print(np.squeeze(Xae_emb3[0,...,0]))
            # print('Reconstructed embeddings from utterance AE')
            # print(np.squeeze(Xae_emb4[0,...,0]))
            # print('Target embeddings')
            # print(np.squeeze(Yae_utt[0,...,0]))
            # raw_input()

            if verbose:
                print('Fitting utterance auto-encoder network')
            self.utt.fit(Xae_utt,
                         Yae_utt,
                         shuffle=True,
                         batch_size=batch_size,
                         epochs=1*nEpoch, # Utt AE gets fewer training samples than Phon AE, so we train more
                         verbose=int(verbose))

        if fitFull:
            if verbose:
                print('Fitting full auto-encoder network')
            self.full.fit(Xae_full,
                          Yae_full,
                          batch_size,
                          shuffle=True,
                          epochs=nEpoch,
                          verbose=int(verbose))

        if verbose:
            print('Evaluating full auto-encoder network')
        eval = self.evaluate(Xae_full, Yae_full, batch_size, verbose=int(verbose))

        if verbose:
            print('Full AE network loss: %.4f' % eval[0])
            if len(eval) > 1:
                print('Full AE network accuracy: %.4f' % eval[1])

        # if verbose:
        #     print('Evaluating full auto-encoder network')
        # eval = self.evaluateA(Xae_full, Yae_full, batch_size, verbose=int(verbose))
        #
        # if verbose:
        #     print('Full AE network loss: %.4f' % eval[0])
        #     if len(eval) > 1:
        #         print('Full AE network accuracy: %.4f' % eval[1])

        return Xae_full, deletedChars_full, oneLetter_full, eval

    def trainFullOnFixed(self, Xs, Xs_mask, segs, maxUtt, maxLen, reverseUtt=False, batch_size=128, nResample=None,
                         fitParts=True, fitFull=True):
        print('Training full auto-encoder network')
        return self.update(Xs,
                           Xs_mask,
                           segs,
                           maxUtt,
                           maxLen,
                           reverseUtt=reverseUtt,
                           batch_size=batch_size,
                           nResample=nResample,
                           fitParts=fitParts,
                           fitFull=fitFull)

    def trainPhonOnFixed(self, Xs, Xs_mask, segs, maxLen, reverseUtt=False, batch_size=128, nResample = None):
        ## Preprocess input data
        Xae, deletedChars, oneLetter = XsSeg2XaePhon(Xs,
                                                     Xs_mask,
                                                     segs,
                                                     maxLen,
                                                     nResample)

        Yae = getYae(Xae, reverseUtt)

        h = self.phon.fit(Xae,
                          Yae,
                          shuffle=True,
                          batch_size=batch_size,
                          epochs=1)
        eval = [h.history[x][0] for x in h.history]

        return Xae, deletedChars, oneLetter, eval

    def plotFull(self, Xae, Yae, logdir, prefix, iteration, batch_size=128, Xae_resamp=None, debug=False):
        ## Initialize plotting objects
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        inputs_src_raw = Xae
        if Xae_resamp is not None:
            ax_input = fig.add_subplot(411)
            ax_input_resamp = fig.add_subplot(412)
            ax_targ = fig.add_subplot(413)
            ax_pred = fig.add_subplot(414)
            inputs_resamp_raw = Xae_resamp
            preds_raw = self.predict(inputs_resamp_raw, batch_size=batch_size)
            if inputs_resamp_raw.shape[-1] == 1:
                inputs_resamp_raw = oneHot(inputs_resamp_raw, preds_raw.shape[-1])
        else:
            ax_input = fig.add_subplot(311)
            ax_targ = fig.add_subplot(312)
            ax_pred = fig.add_subplot(313)
            inputs_src_raw = Xae
            preds_raw = self.predict(inputs_src_raw, batch_size=batch_size)
        targs_raw = Yae
        if inputs_src_raw.shape[-1] == 1:
            inputs_src_raw = oneHot(inputs_src_raw, preds_raw.shape[-1])
        if targs_raw.shape[-1] == 1:
            targs_raw = oneHot(targs_raw, preds_raw.shape[-1])

        for u in range(len(Xae)):
            ## Set up plotting canvas
            fig.patch.set_visible(False)
            fig.suptitle('Utterance %d, Checkpoint %d' % (u, iteration))

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
            fig.savefig(logdir + '/heatmap_' + prefix + '_utt' + str(u) + '_iter' + str(iteration) + '.png')

        plt.close(fig)

    def plotPhon(self, Xae, Yae, logdir, prefix, iteration, batch_size=128, Xae_resamp=None, debug=False):
        ## Initialize plotting objects
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        inputs_src_raw = Xae
        if not Xae_resamp is None:
            ax_input = fig.add_subplot(411)
            ax_input_resamp = fig.add_subplot(412)
            ax_targ = fig.add_subplot(413)
            ax_pred = fig.add_subplot(414)
            inputs_resamp_raw = Xae_resamp
            preds_raw = self.phon.predict(inputs_resamp_raw, batch_size=batch_size)
            if inputs_resamp_raw.shape[-1] == 1:
                inputs_resamp_raw = oneHot(inputs_resamp_raw, preds_raw.shape[-1])
        else:
            ax_input = fig.add_subplot(311)
            ax_targ = fig.add_subplot(312)
            ax_pred = fig.add_subplot(313)
            preds_raw = self.phon.predict(inputs_src_raw, batch_size=batch_size)
        targs_raw = Yae
        if inputs_src_raw.shape[-1] == 1:
            inputs_src_raw = oneHot(inputs_src_raw, preds_raw.shape[-1])
        if targs_raw.shape[-1] == 1:
            targs_raw = oneHot(targs_raw, preds_raw.shape[-1])

        if debug:
            print('=' * 50)
            print('Segmentation details for 10 randomly-selected utterances')
        for w in range(len(Xae)):
            ## Set up plotting canvas
            fig.patch.set_visible(False)
            fig.suptitle('Word %d, Checkpoint %d' % (w, iteration))

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
            fig.savefig(logdir + '/heatmap_' + prefix + '_wrd' + str(w) + '_iter' + str(iteration) + '.png')

        plt.close(fig)

    def plotVAEpyplot(self, logdir, prefix, ctable=None, reverseUtt=False, batch_size=128, debug=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ticks = [[-1,-0.5,0,0.5,1]]*self.latentDim
        samplePoints = np.array(np.meshgrid(*ticks)).T.reshape(-1,3)
        input_placeholder = np.ones(tuple([len(samplePoints)] + list(self.phon.output_shape[1:-1]) + [1]))
        preds = self.decode_word([samplePoints, input_placeholder], batch_size=batch_size)
        if reverseUtt:
            preds = getYae(preds, reverseUtt)
        reconstructed = reconstructXae(np.expand_dims(preds.argmax(-1), -1), ctable, maxLen=5)
        for i in range(len(samplePoints)):
            ax.text(samplePoints[i,0], samplePoints[i,1], samplePoints[i,2], reconstructed[i])
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        pickle.dump(fig, file(logdir + '/' + prefix + '_VAEplot.3D.obj', 'wb'))

        plt.close(fig)

    def plotVAEplotly(self, logdir, prefix, ctable=None, reverseUtt=False, batch_size=128, debug=False):
        ticks = [[-1,-0.5,0,0.5,1]]*self.latentDim
        samplePoints = np.array(np.meshgrid(*ticks)).T.reshape(-1,3)
        input_placeholder = np.ones(tuple([len(samplePoints)] + list(self.phon.output_shape[1:-1]) + [1]))
        preds = self.decode_word([samplePoints, input_placeholder], batch_size=batch_size)
        if reverseUtt:
            preds = getYae(preds, reverseUtt)
        reconstructed = reconstructXae(np.expand_dims(preds.argmax(-1), -1), ctable, maxLen=5)

        data = [go.Scatter3d(
            x = samplePoints[:,0],
            y = samplePoints[:,1],
            z = samplePoints[:,2],
            text = reconstructed,
            mode='text'
        )]
        layout = go.Layout()
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename=logdir + '/' + prefix + '_VAEplot.html', auto_open=False)

    def plotVAE(self, logdir, prefix, ctable=None, reverseUtt=False, batch_size=128, debug=False):
        if usePlotly:
            self.plotVAEplotly(logdir, prefix, ctable, reverseUtt, batch_size, debug)
        else:
            self.plotVAEpyplot(logdir, prefix, ctable, reverseUtt, batch_size, debug)

    def save(self, path, suffix=''):
        full_path = path+'/ae'+suffix+'.h5'
        self.full.save(full_path)
        if self.fitParts:
            phonOptimPath = path + '/ae_phon_optim' + suffix + '.h5'
            save_optimizer_weights(self.phon, phonOptimPath)
            uttOptimPath = path + '/ae_utt_optim' + suffix + '.h5'
            save_optimizer_weights(self.utt, uttOptimPath)

    def load(self, path, suffix=''):
        fullPath = path+'/ae'+suffix+'.h5'
        success = True
        if os.path.exists(fullPath):
            self.full.load_weights(fullPath, by_name=True)
            if self.fitFull:
                load_optimizer_weights(self.full, fullPath)
            if self.fitParts:
                phonOptimPath = path+'/ae_phon_optim'+suffix+'.h5'
                if os.path.exists(phonOptimPath):
                    print('Phon AE optimizer checkpoint found. Loading weights.')
                    load_optimizer_weights(self.phon, phonOptimPath)
                else:
                    print('No phon AE optimizer checkpoint found. Using default initialization.')
                    success = False
                uttOptimPath = path+'/ae_utt_optim'+suffix+'.h5'
                if os.path.exists(uttOptimPath):
                    print('Utt AE optimizer checkpoint found. Loading weights.')
                    load_optimizer_weights(self.utt, uttOptimPath)
                else:
                    print('No utt AE optimizer checkpoint found. Using default initialization.')
                    success = False
        else:
            print('No AE checkpoint found. Using default initialization.')
            success = False

        return success



class Segmenter(object):
    def __init__(self, segmenter, seg_shift=0, charDim=None):
        self.network = segmenter
        self.seg_shift = seg_shift
        self.charDim = charDim

    def update(self, Xs, Xs_mask, targets, batch_size=128):
        seg_shift = self.seg_shift
        charDim = Xs.shape[-1]
        maxChar = Xs_mask.shape[-1]
        seg_inputs = np.zeros((len(Xs), maxChar + seg_shift, charDim))
        seg_inputs[:, :maxChar, :] = Xs
        seg_mask = np.zeros((len(Xs_mask), maxChar + seg_shift))
        seg_mask[:, seg_shift:] = Xs_mask
        seg_mask = np.expand_dims(seg_mask, -1)
        seg_targets = np.zeros((len(targets), maxChar + seg_shift, 1))
        seg_targets[:, seg_shift:, :] = targets
        segHist = self.network.fit([seg_inputs, seg_mask],
                                   seg_targets,
                                   batch_size=batch_size,
                                   epochs=1)
        return segHist

    def predict(self, Xs, Xs_mask, batch_size=128):
        seg_shift = self.seg_shift
        charDim = Xs.shape[-1]
        maxChar = Xs_mask.shape[-1]
        seg_inputs = np.zeros((len(Xs), maxChar + seg_shift, charDim))
        seg_inputs[:, :maxChar, :] = Xs
        seg_mask = np.zeros((len(Xs_mask), maxChar + seg_shift))
        seg_mask[:, seg_shift:] = Xs_mask
        seg_mask = np.expand_dims(seg_mask, -1)
        return self.network.predict([seg_inputs, seg_mask], batch_size=batch_size)[:, seg_shift:, :]

    def trainOnFixed(self, Xs, Xs_mask, Y, batch_size=128):
        print('Training segmenter network')
        self.update(Xs,
                    Xs_mask,
                    Y,
                    batch_size=batch_size)

    def plot(self, Xs, Xs_mask, Y, logdir, prefix, iteration, batch_size=128):
        ## Initialize plotting objects
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        ax_input = fig.add_subplot(311)
        ax_targ = fig.add_subplot(312)
        ax_pred = fig.add_subplot(313)

        inputs_raw = Xs
        masks_raw = Xs_mask
        preds_raw = self.predict(inputs_raw,
                                 masks_raw,
                                 batch_size)

        if inputs_raw.shape[-1] == 1 and self.charDim:
            inputs_raw = oneHot(inputs_raw, self.charDim)
        targs_raw = np.expand_dims(Y, -1)

        for u in range(len(Xs)):
            ## Set up plotting canvas
            fig.patch.set_visible(False)
            fig.suptitle('Utterance %d, Checkpoint %d' % (u, iteration))

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
            fig.savefig(logdir + '/barchart_' + prefix + '_utt' + str(u) + '_iter' + str(iteration) + '.png')

        plt.close(fig)

    def save(self, path, suffix=''):
        path = path+'/seg'+suffix+'.h5'
        self.network.save(path)

    def load(self, path, suffix=''):
        path = path+'/seg'+suffix+'.h5'
        if os.path.exists(path):
            print('Segmenter checkpoint found. Loading weights.')
            self.network.load_weights(path, by_name=True)
            load_optimizer_weights(self.network, path)
            return True
        else:
            print('No segmenter checkpoint found. Using default initialization.')
            return False





# #################################################################################
# #################################################################################
# #
# #  Network Graph Construction
# #
# #################################################################################
# #################################################################################
#
# ## TODO: Weight saving/loading doesn't work for some reason when the networks are initialized this way
#
# def constructNetworks(wordHidden, uttHidden, charDropout, wordDropout, maxChar, maxUtt, maxLen, charDim, segHidden, sess, logdir,
#                       aeNet=False, aeType=None, vae=False, latentDim=None, segNet=False, segShift=0, optim='nadam',
#                       reverseUtt=False, nResample=None, acoustic=False, loadModels=True):
#     print("Constructing networks")
#
#     K.set_session(sess)
#
#     RNN = recurrent.LSTM
#
#     adam = optimizers.adam(clipnorm=1.)
#     nadam = optimizers.Nadam(clipnorm=1.)
#     rmsprop = optimizers.RMSprop(clipnorm=1.)
#     optim_map = {'adam': adam, 'nadam': nadam, 'rmsprop': rmsprop}
#
#     if aeNet:
#         ## AUTO-ENCODER NETWORK
#         print('Using %s for auto-encoding' % (aeType.upper()))
#         print()
#
#         ## USEFUL VARIABLES
#         wLen = nResample if nResample else maxLen
#         wEmbDim = latentDim if vae else wordHidden
#         if aeType == 'cnn':
#             x_wrd_pad = int(2 ** (math.ceil(math.log(wLen, 2))) - wLen)
#             y_wrd_pad = int(2 ** (math.ceil(math.log(charDim, 2))) - charDim)
#             x_utt_pad = int(2 ** (math.ceil(math.log(maxUtt, 2))) - maxUtt)
#             y_utt_pad = int(2 ** (math.ceil(math.log(wEmbDim, 2))) - wEmbDim)
#             expand = Lambda(lambda x: K.expand_dims(x, -1), name='ExpandDim')
#             squeeze = Lambda(lambda x: K.squeeze(x, -1), name='SqueezeDim')
#             nFilters = 10
#
#         ## INPUTS
#         fullInput = Input(shape=(maxUtt, wLen, charDim), name='FullInput')
#         phonInput = Input(shape=(wLen, charDim), name='PhonInput')
#         uttInput = Input(shape=(maxUtt, latentDim if vae else wordHidden), name='UttInput')
#         wordDecInput = Input(shape=(latentDim if vae else wordHidden,), name='WordDecoderInput')
#         uttDecInput = Input(shape=(uttHidden,), name='UttDecoderInput')
#
#         ## OUTPUT MASKS
#         m_phon2phon = (lambda x: x * K.cast(K.any(K.reverse(phonInput, 1), -1, keepdims=True), 'float32')) if reverseUtt else \
#                       (lambda x: x * K.cast(K.any(phonInput, -1, keepdims=True), 'float32'))
#         m_utt2utt =   (lambda x: x * K.cast(K.any(K.reverse(uttInput, 1), -1, keepdims=True), 'float32')) if reverseUtt else \
#                       (lambda x: x * K.cast(K.any(uttInput, -1, keepdims=True), 'float32'))
#         m_full2utt =  lambda x: x * K.cast(K.any(K.any(fullInput, -1), -1, keepdims=True), 'float32')
#         m_full2uttR = (lambda x: x * K.cast(K.any(K.any(K.reverse(fullInput, (1, 2)), -1), -1, keepdims=True), 'float32')) if reverseUtt else \
#                       (lambda x: x * K.cast(K.any(K.any(fullInput, -1), -1, keepdims=True), 'float32'))
#         m_full2full = (lambda x: x * K.cast(K.any(K.reverse(fullInput, (1, 2)), -1, keepdims=True), 'float32')) if reverseUtt else \
#                       (lambda x: x * K.cast(K.any(fullInput, -1, keepdims=True), 'float32'))
#
#         ## WORD ENCODER
#         wordEncoder = Masking(mask_value=0.0, name='WordEncoderInputMask')(phonInput)
#         wordEncoder = Dropout(charDropout, noise_shape=(1, wLen, 1), name='CharacterDropout')(wordEncoder)
#         if aeType == 'rnn':
#             wordEncoder = RNN(wordHidden, name='WordEncoderRNN')(wordEncoder)
#         elif aeType == 'cnn':
#             wordEncoder = expand(wordEncoder)
#             wordEncoder = ZeroPadding2D(padding=((0, x_wrd_pad), (0, y_wrd_pad)), name='WordInputPadder')(wordEncoder)
#             wordEncoder = Conv2D(nFilters, (3, 3), padding='same', activation='elu', name='WordConv')(wordEncoder)
#             wordEncoder = Conv2D(nFilters, (3, 3), strides=(2,2), padding='same', activation='elu', name='WordStrideConv')(wordEncoder)
#             wordEncoder = Flatten(name='WordConvFlattener')(wordEncoder)
#             wordEncoder = Dense(wordHidden, activation='elu', name='WordFullyConnected')(wordEncoder)
#         if vae:
#             word_mean = Dense(latentDim, name='WordMean')(wordEncoder)
#             word_log_var = Dense(latentDim, name='WordVariance')(wordEncoder)
#
#             def sampling(args):
#                 word_mean, word_log_var = args
#                 epsilon = K.random_normal(shape=K.shape(word_mean), mean=0., stddev=1.0)
#                 return word_mean + K.exp(word_log_var/2) * epsilon
#             wordEncoder = Lambda(sampling, output_shape=(latentDim,), name='WordEmbSampler')([word_mean, word_log_var])
#         elif aeType == 'cnn':
#             wordEncoder = Dense(wordHidden, name='WordEncoderOut')(wordEncoder)
#         wordEncoder = Model(inputs=phonInput, outputs=wordEncoder, name='WordEncoder')
#
#         ## WORDS ENCODER
#         wordsEncoder = TimeDistributed(wordEncoder, name='WordEncoderDistributer')(fullInput)
#         wordsEncoder = Lambda(m_full2utt, name='WordsEncoderPremask')(wordsEncoder)
#         wordsEncoder = Model(inputs=fullInput, outputs=wordsEncoder, name='WordsEncoder')
#
#         ## UTTERANCE ENCODER
#         uttEncoder = Masking(mask_value=0.0, name='UttInputMask')(uttInput)
#         uttEncoder = Dropout(wordDropout, noise_shape=(1, maxUtt, 1), name='WordDropout')(uttEncoder)
#         if aeType == 'rnn':
#             uttEncoder = RNN(uttHidden, return_sequences=False, name='UttEncoderRNN')(uttEncoder)
#         elif aeType == 'cnn':
#             uttEncoder = expand(uttEncoder)
#             uttEncoder = ZeroPadding2D(padding=((0, x_utt_pad), (0, y_utt_pad)), name='UttInputPadder')(uttEncoder)
#             uttEncoder = Conv2D(nFilters, (3, 3), padding='same', activation='elu', name='UttConv')(uttEncoder)
#             uttEncoder = Conv2D(nFilters, (3, 3), strides=(2, 2), padding='same', activation='elu', name='UttStrideConv')(uttEncoder)
#             uttEncoder = Flatten(name='UttConvFlattener')(uttEncoder)
#             uttEncoder = Dense(uttHidden, activation='elu', name='UttFullyConnected')(uttEncoder)
#             uttEncoder = Dense(uttHidden, name='UttEncoderOut')(uttEncoder)
#         uttEncoder = Model(inputs=uttInput, outputs=uttEncoder, name='UttEncoder')
#
#         ## UTTERANCE DECODER
#         if aeType == 'rnn':
#             uttDecoder = RepeatVector(maxUtt, input_shape=(uttHidden,), name='UttEmbeddingRepeater')(uttDecInput)
#             uttDecoder = RNN(uttHidden, return_sequences=True, name='UttDecoderRNN')(uttDecoder)
#             uttDecoder = TimeDistributed(Dense(wEmbDim), name='UttDecoderOut')(uttDecoder)
#         elif aeType == 'cnn':
#             uttDecoder = Dense(int((maxUtt + x_utt_pad) / 2 * (wEmbDim + y_utt_pad) / 2 * nFilters), activation='elu', name='UttDecoderDenseIn')(uttDecInput)
#             uttDecoder = Reshape((int((maxUtt + x_utt_pad) / 2), int((wEmbDim + y_utt_pad) / 2), nFilters), name='UttDecoderReshape')(uttDecoder)
#             uttDecoder = Conv2D(nFilters, (3, 3), padding='same', activation='elu', name='UttDeconv')(uttDecoder)
#             uttDecoder = UpSampling2D((2, 2), name='UttUpsample')(uttDecoder)
#             uttDecoder = Cropping2D(((0, x_utt_pad), (0, y_utt_pad)), name='UttOutCrop')(uttDecoder)
#             uttDecoder = Conv2D(1, (3, 3), padding='same', activation='linear', name='UttDecoderOut')(uttDecoder)
#             uttDecoder = squeeze(uttDecoder)
#         uttDecoder = Model(inputs=uttDecInput, outputs=uttDecoder, name='UttDecoder')
#
#         ## WORD DECODER
#         if aeType == 'rnn':
#             wordDecoder = RepeatVector(wLen, input_shape=(wEmbDim,), name='WordEmbeddingRepeater')(wordDecInput)
#             wordDecoder = Masking(mask_value=0, name='WordDecoderInputMask')(wordDecoder)
#             wordDecoder = RNN(wordHidden, return_sequences=True, name='WordDecoderRNN')(wordDecoder)
#             wordDecoder = TimeDistributed(Dense(charDim), name='WordDecoderDistributer')(wordDecoder)
#             wordDecoder = Activation('linear' if acoustic else 'softmax', name='WordDecoderOut')(wordDecoder)
#         elif aeType == 'cnn':
#             wordDecoder = Dense(int((wLen + x_wrd_pad) / 2 * (charDim + y_wrd_pad) / 2 * nFilters), activation='elu', name='WordDecoderDenseIn')(wordDecInput)
#             wordDecoder = Reshape((int((wLen + x_wrd_pad) / 2), int((charDim + y_wrd_pad) / 2), nFilters), name='WordDecoderReshape')(wordDecoder)
#             wordDecoder = Conv2D(nFilters, (3, 3), padding='same', activation='elu', name='WordDeconv')(wordDecoder)
#             wordDecoder = UpSampling2D((2,2), name='WordUpsample')(wordDecoder)
#             wordDecoder = Cropping2D(((0, x_wrd_pad), (0, y_wrd_pad)), name='WordOutCrop')(wordDecoder)
#             wordDecoder = Conv2D(1, (3, 3), padding='same', activation='linear' if acoustic else 'softmax', name='WordDecoderOut')(wordDecoder)
#             wordDecoder = squeeze(wordDecoder)
#         wordDecoder = Model(inputs=wordDecInput, outputs=wordDecoder, name='WordDecoder')
#
#         ## WORDS DECODER (OUTPUT LAYER)
#         wordsDecoder = TimeDistributed(wordDecoder, name='WordsDecoderDistributer')(uttInput)
#         wordsDecoder = Masking(mask_value=0.0, name='WordsDecoderInputMask')(wordsDecoder)
#         wordsDecoder = Model(inputs=uttInput, outputs=wordsDecoder, name='WordsDecoder')
#
#         ## ENCODER-DECODER LAYERS
#         wordEncoderTensor = wordEncoder(phonInput)
#         wordsEncoderTensor = wordsEncoder(fullInput)
#         wordDecoderTensor = wordDecoder(wordDecInput)
#         wordsDecoderTensor = wordsDecoder(uttInput)
#
#         phonEncoderDecoder = wordDecoder(wordEncoderTensor)
#         phonEncoderDecoder = Lambda(m_phon2phon, name='PhonPremask')(phonEncoderDecoder)
#
#         uttEncoderDecoder = uttDecoder(uttEncoder(uttInput))
#         uttEncoderDecoder = Model(inputs=uttInput, outputs=uttEncoderDecoder, name='UttEncoderDecoder')
#
#         fullEncoderUttDecoder = uttEncoderDecoder(wordsEncoderTensor)
#         fullEncoderUttDecoder = Lambda(m_full2uttR, name='UttPremask')(fullEncoderUttDecoder)
#
#         fullEncoderDecoder = wordsDecoder(fullEncoderUttDecoder)
#         fullEncoderDecoder = Lambda(m_full2full, name='FullPremask')(fullEncoderDecoder)
#
#         ## VAE LOSS
#         if vae:
#             def vae_loss(y_true, y_pred):
#                 loss_func = metrics.mean_squared_error if acoustic else masked_categorical_crossentropy
#                 ae_loss = loss_func(y_true, y_pred)
#                 ## We keep dims to tile the kl_loss out to all reconstructed characters/frames
#                 kl_loss = - 0.5 * K.mean(1 + word_log_var - K.square(word_mean) - K.exp(word_log_var), axis=-1,
#                                          keepdims=True)
#                 return ae_loss + kl_loss
#
#         ## COMPILED (TRAINABLE) MODELS
#         ae_phon = Masking(mask_value=0, name='PhonMask')(phonEncoderDecoder)
#         ae_phon = Model(inputs=phonInput, outputs=ae_phon, name='AEPhon')
#         ae_phon.compile(
#             loss=vae_loss if vae else "mean_squared_error" if acoustic else masked_categorical_crossentropy,
#             metrics=None if acoustic else [masked_categorical_accuracy],
#             optimizer=optim_map[optim])
#
#         ae_utt = Masking(mask_value=0, name='UttMask')(Lambda(m_utt2utt, name='UttPremask')(uttEncoderDecoder(uttInput)))
#         ae_utt = Model(inputs=uttInput, outputs=ae_utt, name='AEUtt')
#         ae_utt.compile(loss="mean_squared_error", optimizer=optim_map[optim])
#
#         ae_full = Masking(mask_value=0, name='AEFullMask')(fullEncoderDecoder)
#         ae_full = Model(inputs=fullInput, outputs=ae_full, name='AEFull')
#         ae_full.compile(loss="mean_squared_error" if acoustic else masked_categorical_crossentropy,
#                         metrics=None if acoustic else [masked_categorical_accuracy],
#                         optimizer=optim_map[optim])
#
#         ## EMBEDDING/DECODING FEEDFORWARD SUB-NETWORKS
#         ## Embeddings must be custom Keras functions instead of models to allow predict with and without dropout
#         embed_word = K.function(inputs=[phonInput, K.learning_phase()], outputs=[wordEncoderTensor], name='EmbedWord')
#         embed_word = makeFunction(embed_word)
#
#         embed_words = K.function(inputs=[fullInput, K.learning_phase()], outputs=[wordsEncoderTensor], name='EmbedWords')
#         embed_words = makeFunction(embed_words)
#
#         embed_words_reconst = K.function(inputs=[fullInput, K.learning_phase()], outputs=[fullEncoderUttDecoder], name='EmbedWordsReconstructed')
#         embed_words_reconst = makeFunction(embed_words_reconst)
#
#         word_decoder = Model(inputs=wordDecInput, outputs=wordDecoderTensor, name='WordDecoder')
#         word_decoder.compile(loss="mean_squared_error" if acoustic else masked_categorical_crossentropy,
#                              metrics=None if acoustic else [masked_categorical_accuracy],
#                              optimizer=optim_map[optim])
#
#         words_decoder = Model(inputs=uttInput, outputs=wordsDecoderTensor, name='WordsDecoder')
#         words_decoder.compile(loss="mean_squared_error" if acoustic else masked_categorical_crossentropy,
#                               metrics=None if acoustic else [masked_categorical_accuracy],
#                               optimizer=optim_map[optim])
#
#         ## (SUB-)NETWORK SUMMARIES
#         print('\n')
#         print('='*50)
#         print('(Sub-)Model Summaries:')
#         print('='*50)
#         print('\n')
#         print('Word encoder model:')
#         wordEncoder.summary()
#         print('\n')
#         print('Word decoder model:')
#         wordDecoder.summary()
#         print('\n')
#         print('Utterance encoder model:')
#         uttEncoder.summary()
#         print('\n')
#         print('Utterance decoder model:')
#         uttDecoder.summary()
#         print('\n')
#         print('Phonological auto-encoder model:')
#         ae_phon.summary()
#         print('\n')
#         print('Utterance auto-encoder model:')
#         ae_utt.summary()
#         print('\n')
#         print('Full auto-encoder model:')
#         ae_full.summary()
#         print('\n')
#
#         ## Initialize AE wrapper object containing all sub nets for convenience
#         ae = AE(ae_full, ae_utt, ae_phon, embed_word, embed_words, embed_words_reconst, word_decoder, words_decoder)
#
#     if segNet:
#         ## SEGMENTER NETWORK
#         segInput = Input(shape=(maxChar + segShift, charDim), name='SegmenterInput')
#         segMaskInput = Input(shape=(maxChar + segShift, 1), name='SegmenterMaskInput')
#
#         segmenter = Sequential(name="Segmenter")
#         segmenter.add(RNN(segHidden, return_sequences=True, input_shape=(maxChar + segShift, charDim)))
#         segmenter.add(TimeDistributed(Dense(1)))
#         segmenter.add(Activation("sigmoid"))
#         segmenterPremask = Lambda(lambda x: x[0] * (1- K.cast(x[1], 'float32')), name='Seg-output-premask')([segmenter(segInput), segMaskInput])
#         segmenter = Masking(mask_value=0.0, name='Seg-mask')(segmenterPremask)
#         segmenter = Model(inputs=[segInput, segMaskInput], outputs= segmenter)
#         segmenter.compile(loss="binary_crossentropy",
#                           optimizer=optim_map[optim])
#         print('Segmenter network summary')
#         segmenter.summary()
#         print('')
#
#         ## Initialize Segmenter wrapper object for convenience
#         segmenter = Segmenter(segmenter, segShift)
#
#     return ae if aeNet else None, segmenter if segNet else None





##################################################################################
##################################################################################
##
##  Cross-Validation Evaluation
##
##################################################################################
##################################################################################

def evalCrossVal(Xs, Xs_mask, gold, doc_list, doc_indices, utt_ids, otherParams, maxLen, maxUtt, raw_total, logdir,
                 iteration, batch_num, reverseUtt=False, batch_size=128, nResample=None, acoustic=False, ae=None,
                 segmenter=None, debug=False):
    if acoustic:
        vadIntervals, GOLDWRD, GOLDPHN, vad = otherParams
    else:
        ctable = otherParams
    ae_net = ae != None
    seg_net = segmenter != None
    print()
    print('*'*50)
    print('Performing system evaluation (cross-validation set)')
    print('Using segmentations predicted by the segmenter network')
    print('Segmenting data')
    if seg_net:
        preds = segmenter.predict(Xs,
                                  Xs_mask,
                                  batch_size)
        segs4eval = pSegs2Segs(preds, acoustic)
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
        print('Computing network losses')
        eval = ae.evaluate(Xae, Yae, batch_size=batch_size)
        cvAELoss = eval[0]
        if not acoustic:
            cvAEAcc = eval[1]
        cvDel = deletedChars.sum()
        cvOneL = oneLetter.sum()

        if not acoustic:
            print('')
            print('Example reconstruction of learned segmentation')
            printReconstruction(utt_ids, ae, Xae, ctable, batch_size, reverseUtt)

    segs4evalXDoc = dict.fromkeys(doc_list)
    for doc in segs4evalXDoc:
        s, e = doc_indices[doc]
        segs4evalXDoc[doc] = segs4eval[s:e]
        if acoustic:
            masked_proposal = np.ma.array(segs4evalXDoc[doc], mask=Xs_mask[s:e])
            segs4evalXDoc[doc] = masked_proposal.compressed()

    print('Scoring segmentation')
    segScore = writeLog(batch_num,
                        iteration,
                        cvAELoss if ae_net else None,
                        cvAEAcc if not acoustic else None,
                        None,
                        cvDel if ae_net else None,
                        cvOneL if ae_net else None,
                        cvSeg,
                        gold,
                        segs4evalXDoc,
                        logdir,
                        vadIntervals= vadIntervals if acoustic else None,
                        acoustic = acoustic,
                        print_headers = not os.path.isfile(logdir + '/log_cv.txt'),
                        filename = 'log_cv.txt')

    print('Total frames:', raw_total)
    if ae_net:
        print('Auto-encoder loss:', cvAELoss)
        if not acoustic:
            print('Auto-encoder accuracy:', cvAEAcc)
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
        writeTimeSegs(segs2Intervals(segs4evalXDoc, vadIntervals), out_dir=logdir, TextGrid=False, dataset='cv')
        writeTimeSegs(segs2Intervals(segs4evalXDoc, vadIntervals), out_dir=logdir, TextGrid=True, dataset='cv')
    else:
        print('Writing solutions to file')
        printSegScores(getSegScores(gold, segs4evalXDoc, acoustic), acoustic)
        # writeCharSegs(logdir, segs4evalXDoc[doc_list[0]], gold[doc_list[0]], batch_num, filename='seg_cv.txt')

    print()
    print('Plotting visualizations on cross-validation set')
    if ae_net:
        if nResample:
            Xae_full, _, _ = XsSeg2Xae(Xs[utt_ids],
                                       Xs_mask[utt_ids],
                                       segs4eval[utt_ids],
                                       maxUtt,
                                       maxLen,
                                       nResample=None)

        ae.plotFull(Xae_full if nResample else Xae[utt_ids],
                    Yae[utt_ids],
                    logdir,
                    'cv',
                    batch_num,
                    batch_size=batch_size,
                    Xae_resamp = Xae[utt_ids] if nResample else None,
                    debug=debug)

    segmenter.plot(Xs[utt_ids],
                   Xs_mask[utt_ids],
                   segs4eval[utt_ids],
                   logdir,
                   'cv',
                   batch_num,
                   batch_size=batch_size)
    print('*' * 50)
