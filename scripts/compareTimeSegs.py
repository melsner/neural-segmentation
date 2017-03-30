import sys, argparse
from score import *

argparser = argparse.ArgumentParser()
argparser.add_argument('gold')
argparser.add_argument('test')
args = argparser.parse_args()

gold = readGoldFrameSeg(args.gold)
test = readGoldFrameSeg(args.test)

scores = getSegScore(gold, test, True)
printSegScore(scores, True)
