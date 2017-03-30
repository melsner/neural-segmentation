import sys, os

with open('user-lexicondiscovery-directory.txt', 'rb') as ldpath:
    lddir = ldpath.readline().strip()

wordseg = []
with open(lddir + '/exps/18.06-1999/L02/global.sec.wrd') as wordsegpath:
    for line in wordsegpath:
        line = line.strip().split()
        if len(line) > 2:
            word = [float(line[0]), float(line[1]), line[2]]
            wordseg.append(word)
        else:
            word = [float(line[0]), float(line[1]), None]
            if len(wordseg) > 0 and wordseg[-1][2] == None:
                wordseg[-1][1] = word[1]
            else:
                wordseg.append(word)

exp1dir = lddir + '/exps/18.06-1999/L02/data/discovered_units/'

pluseg = [[0.0,0.0,[]]]
word_ix = 0
word_end = 0.0
clock = 0.0
filelist = [exp1dir + x for x in os.listdir(exp1dir) if x.endswith('.algn')]
filelist.sort()

for x in filelist:
    print(x)
    with open(x, 'rb') as f:
        for line in f:
            plu = line.strip().split()
            start = clock + float(plu[0])/100
            end = clock + (float(plu[1])+1)/100
            cluster_id = int(plu[2])

            print(start)
            print(end)
            print(cluster_id)
            print(wordseg[word_ix][2])
            print('')
            if end > wordseg[word_ix][1]:
                word_ix += 1
                if len(pluseg) > 1 and abs(end-word_end) > abs(word_end-pluseg[-1][1]):
                    # PLU starts a new word
                    pluseg.append([start,end,[cluster_id]])
                else:
                    # PLU finishes the last word
                    pluseg[-1][1] = end
                    pluseg[-1][2].append(cluster_id)
                    pluseg.append([end, end, []])
            else:
                pluseg[-1][1] = end
                pluseg[-1][2].append(cluster_id)
    clock = end

#for x in pluseg:
#    print(x)

