import sys, os

path = "logs/"
dirlist = os.listdir(path)

getHeader = True
header = []
rows = []

for logdir in dirlist:
    logfile = path+logdir+'/log.txt'
    with open(logfile, 'rb') as log:
        h = ['name'] + log.readline().strip().split()
        if getHeader:
            header = h
            getHeader = False
        lines = log.readlines()
        logrow = [logdir] + lines[-1].strip().split()
        rows.append(logrow)

print(' '.join(header))
for row in rows:
    print(' '.join(row))
