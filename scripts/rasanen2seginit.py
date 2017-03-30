import sys

docs = {}

for line in sys.stdin:
    if line.startswith('s'):
        doc, start, end = line.split()
        if doc in docs:
            docs[doc].append((doc, float(start),float(end)))
        else:
           docs[doc] = [(doc, float(start),float(end))]

for doc in docs:
    docs[doc].sort(key=lambda x: x[1])

for doc in sorted(docs.keys()):
    for p in docs[doc]:
        print('%s %s %s' %p)
    print('')
