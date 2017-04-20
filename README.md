# neural-segmentation

The main file is /scripts/autoencodeDecodeChars.py

For Brent, use:

--uttHidden 400 --wordHidden 40 --wordDropout 0.25 --charDropout 0.5 br-phono.txt

For Zerospeech, use:

--acoustic --uttHidden 400 --wordHidden 20 --wordDropout 0.25 --charDropout 0 --segHidden 1000 --segFile [path-to-vad] [path-to-mfccs]

The remaining arguments will be set to sensible defaults for acoustic
or character mode.

usage: autoencodeDecodeChars.py [-h] [--uttHidden UTTHIDDEN]
                                [--wordHidden WORDHIDDEN]
                                [--segHidden SEGHIDDEN]
                                [--wordDropout WORDDROPOUT]
                                [--charDropout CHARDROPOUT] [--metric METRIC]
                                [--pretrainIters PRETRAINITERS]
                                [--trainNoSegIters TRAINNOSEGITERS]
                                [--trainIters TRAINITERS] [--maxChar MAXCHAR]
                                [--maxLen MAXLEN] [--maxUtt MAXUTT]
                                [--delWt DELWT] [--oneLWt ONELWT]
                                [--segWt SEGWT] [--nSamples NSAMPLES]
                                [--batchSize BATCHSIZE] [--logfile LOGFILE]
                                [--acoustic] [--segfile SEGFILE]
                                [--goldwrd GOLDWRD] [--goldphn GOLDPHN]
                                [--gpufrac GPUFRAC]
								data

You will need Keras and either Theano or Tensorflow. A GPU is strongly
recommended (use --gpufrac to control your memory usage if you're sharing).

The program automatically logs metrics to --logfile. For Zerospeech,
you need to pass it the gold filepaths automatically; for Brent, it
reads them based on the initial spaces in the input file.

We have not included the MFCCs for Zerospeech due to their size; you
can make them by obtaining the data here:
http://sapience.dec.ens.fr/bootphon/2015/index.html and using Kaldi to
compute them.
