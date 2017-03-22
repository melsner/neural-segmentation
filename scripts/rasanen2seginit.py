import sys

docs = {}

for line in sys.stdin:
    if line.startswith('s'):
        doc, start, end = line.split()
        if doc in docs:
            docs[doc].append((float(start),float(end)))
        else:
           docs[doc] = [(float(start),float(end))]

for doc in docs:
    docs[doc].sort(key=lambda x: x[0])

for doc in sorted(docs.keys()):
    print(doc)
    for p in docs[doc]:
        print('%s %s' %p)
    print('')
