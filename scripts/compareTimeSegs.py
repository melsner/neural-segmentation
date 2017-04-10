from __future__ import print_function, division
import sys, argparse
from score import *
from ae_io import *

argparser = argparse.ArgumentParser()
argparser.add_argument('gold')
argparser.add_argument('test')
args = argparser.parse_args()

gold = readGoldFrameSeg(args.gold)
test = readGoldFrameSeg(args.test)

scores = getSegScores(gold, test, True)
printSegScores(scores, True)
