import sys, os

data = sys.argv[1]
clistr = ' '.join(sys.argv[2:])
argchunks = clistr.split('--')
if argchunks[0] =='':
    argchunks = argchunks[1:]
for i in range(len(argchunks)):
    argchunks[i] = argchunks[i].strip()

args = []

if not os.path.exists('batch'):
    os.makedirs('batch')

def printCallList(args, prefix, logname):
    if len(args) == 0:
        with open('batch/' + logname + '.sh', 'wb') as f: 
            f.write('#PBS -N %s\n' %logname + prefix + ' --logfile ' + logname + '\n')
    else:
        for v in args[0]:
            if len(args[0]) > 1:
                if args[0][0] == '':
                    if v == '':
                        xtralogname = '_no' + args[0][1][2:]
                    else:
                        xtralogname = '_' + v[2:]
                else:
                    xtralogname = '_' + prefix.split()[-1][2:4].upper() + v
            else:
                xtralogname = ''
            printCallList(args[1:], prefix + ' ' + v, logname + xtralogname)

for c in argchunks:
    clist = c.split()
    if len(clist) > 1 and clist[1] == '?':
        args.append(['','--' + clist[0]])
    else:
        args.append(['--' + clist[0]])
    if len(clist) > 1 and clist[1] != '?':
        args.append(clist[1:])

if 'acoustic' in argchunks:
    logname = 'acoustic'
else:
    logname = 'text'

prefix = '''#PBS -l nodes=1:gpus=1
#PBS -l mem=16gb
#PBS -l walltime=72:00:00

module load python/2.7
cd $PBS_O_WORKDIR
source activate keras
export KERAS_BACKEND=tensorflow
module load cuda

python scripts/main.py %s''' %data

printCallList(args, prefix, logname)
