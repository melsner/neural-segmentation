import sys, numpy as np

def readGoldFrameSeg(path): 
    gold_seg = {} 
    with open(path, 'rb') as gold: 
        for line in gold: 
            if line.strip() != '': 
                doc, start, end = line.strip().split()[:3] 
                if doc in gold_seg: 
                    gold_seg[doc].append((float(start),float(end))) 
                else: 
                    gold_seg[doc] = [(float(start),float(end))] 
    for doc in gold_seg: 
        gold_seg[doc].sort(key=lambda x: x[0]) 
    return gold_seg 


