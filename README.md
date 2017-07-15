# Unsupervised RNN speech segmenter

This repository contains code for the unsupervised RNN speech segmentation system
presented in Elsner & Shain (to appear), EMNLP. The head of the branch is under
active development and may contain bugs and/or significant differences from the
version that produced the published result. To reproduce the results in Elsner & Shain,
first checkout the appropriate commit using the following command:

`git checkout cee1de2db85d08fde92562ca81a544a106f74a51`

The results can be reproduced as follows.

For the text (Brent corpus) results, run:

`python scripts/autoencodeDecodeChars.py br-phono.txt --uttHidden 400 --wordHidden 40 --wordDropout 0.25 --charDropout 0.5 --logfile emnlp_text`

For the acoustic (Zerospeech 15) results, run:

`python scripts/autoencodeDecodeChars.py <ENGLISH-MFCC-DIRECTORY> --acoustic --logfile emnlp_acoustic --segfile <ENGLISH-VAD-FILE> --goldphn <ENGLISH-PHN-FILE> --goldwrd <ENGLISH-WRD-FILE> --wordHidden 20 --segHidden 1500 --uttHidden 400 --maxChar 400 --maxLen 100 --maxUtt 16 --batchSize 5000`


To use the system in its current state, use `python scripts/main.py` with
appropriate arguments. For usage details, run `python scripts/main.py -h`.

You will need Keras and either Theano or Tensorflow. A GPU is strongly
recommended (use --gpufrac to control your memory usage if you're sharing).

The program automatically logs metrics to --logfile. For Zerospeech,
you need to pass it the gold filepaths automatically; for Brent, it
reads them based on the initial spaces in the input file.

We have not included the MFCCs for Zerospeech due to their size; you
can make them by obtaining the data here:
http://sapience.dec.ens.fr/bootphon/2015/index.html and using Kaldi to
compute them. The files `<ENGLISH-VAD-FILE>`,
`<ENGLISH-PHN-FILE>` and `<ENGLISH-WRD-FILE>` are `english_vad.txt`, `english.phn`,
and `english.wrd` provided in the challenge data.
