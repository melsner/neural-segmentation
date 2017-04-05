import os

if __name__ == "__main__":
    preamble = file("scripts/runNN.sh").read()

    for wSize in [40]:
        for uSize in [400]:
            for sSize in [500, 1000, 1500]:
                for wDrop in [.25]:
                    for cDrop in [.25]:
                        for maxLen in [50, 100]:
                            for maxChar in [250, 500]:
                                for batchSize in [500,1000,5000]:
                                    rName = "noisy-w%d-u%d-s%d-wd%g-c%g-l%d-C%d-b%d" % (wSize, uSize, 
                                                                       sSize, wDrop, cDrop, maxLen, maxChar, batchSize)
                                    cmd = "python scripts/autoencodeDecodeChars.py build/zerospeech/test/ --acoustic --segfile data/test_vad.txt --goldfile data/test.phn --gpufrac None --wordHidden %d --uttHidden %d --segHidden %d --wordDropout %g --charDropout %g --maxLen %d --maxChar %d --batchSize %d --logfile %s/" % (wSize, uSize, sSize, wDrop, cDrop, maxLen, maxChar, batchSize, rName)

                                    outf = file("batch/run%s.sh" % rName, "w")
                                    print >>outf, preamble + "\n" + cmd
                                    outf.close()
                                    
