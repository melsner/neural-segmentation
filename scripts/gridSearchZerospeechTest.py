import os, sys

if len(sys.argv) > 1:
    run = bool(sys.argv[1])
else:
    run = False

if __name__ == "__main__":
    preamble = file("scripts/runNN.sh").read()

    for wSize in [20, 80]:
        for uSize in [400]:
            for sSize in [1500]:
                for wDrop in [0, .25]:
                    for cDrop in [0, .25]:
                        for maxLen in [100]:
                            for maxChar in [500]:
                                for nSamples in [100]:
                                    for meanWLen in [10,30]:
                                        for batchSize in [1000,5000]:
                                            maxUtt = maxChar/meanWLen
                                            rName = "noisy-w%d-u%d-s%d-wd%g-c%g-l%d-C%d-U%d-b%d" % (wSize, uSize, 
                                                                               sSize, wDrop, cDrop, maxLen, maxChar, maxUtt, batchSize)
                                            cmd = "python scripts/autoencodeDecodeChars.py build/zerospeech/test/ --acoustic --segfile data/test_vad.txt --goldfile data/test.phn --gpufrac None --wordHidden %d --uttHidden %d --segHidden %d --wordDropout %g --charDropout %g --maxLen %d --maxChar %d --maxUtt %d --batchSize %d --logfile %s/" % (wSize, uSize, sSize, wDrop, cDrop, maxLen, maxChar, maxUtt, batchSize, rName)

                                            outf = file("batch/run%s.sh" % rName, "w")
                                            print >>outf, preamble + "\n" + cmd
                                            outf.close()
                                            if run:
                                                os.system("qsub batch/run%s.sh" %rName)
                                            
