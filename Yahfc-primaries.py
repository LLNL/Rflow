#! /usr/bin/env python

import sys,os,re,argparse
from nuclear import *

parser = argparse.ArgumentParser(description='Merge two GNDS libraries, prefering dir1, t')
# Process command line options
parser.add_argument('primary_dir', type=str, help='directory of primary decays from Yahfc')
parser.add_argument("-o", "--out", type=str, help='directory for excitation functions')
parser.add_argument("-u", "--units", type=str, default='mb', help='b or mb or ub: output units')

args = parser.parse_args()

print('\nYAHFC primaries')
cmd = ' '.join([t if '*' not in t else ("'%s'" % t) for t in sys.argv[:]])
print('Command:',cmd ,'\n')

qnames = {0:'g', 1:'n', 1001:'p', 1002:'d',1003:'t', 2003:'h',2004:'a',}
scales = {'b':1, 'mb':1000, 'ub':1e6}

discreteXsec = {}
binXsec = {}
discreteLevels = {}

xs_scale = scales[args.units]

for root,dirs,files in os.walk(args.primary_dir):
    for name in sorted(files):
        if 'swp' in name: continue        
        incident,_,out = name.replace('.dat','').split('_')
        projectile,target = incident.split('+')
        Ein,residual = out.split('-')
        Ein = float(Ein)
        
        ZAp = None
        for za in qnames.keys(): 
            if qnames[za] == projectile: ZAp = za
        Zp = ZAp // 1000
        Ap = ZAp % 1000
        At = int(re.search('(\d+)',target).group())
        St = re.search('(\D+)',target).group()
        Zt = elementZFromSymbol(St)
        Ar = int(re.search('(\d+)',residual).group())
        Sr = re.search('(\D+)',residual).group()
        print('From',name,'projectile:',Zp,Ap,'residual:',residual,Ar,Sr)
        Zr = elementZFromSymbol(Sr)
        Ze = Zt+Zp - Zr
        Ae = At+Ap - Ar
        ZAe = Ze*1000 + Ae
        ejectile = qnames[ZAe]
        print("Target Z,A=",Zt*1000 + At,'to residual',Zr*1000 + Ar,' ejectile=',ZAe,':',ejectile)
        
        if args.out is None:
            args.out = 'yahfc-%s' % incident + '.excitations'

        fullpath =  os.path.join(root,name)
        print('Open %s' % fullpath)
        lines = open(fullpath,'r').readlines()
        for i in range(5,99999,3):
            line = lines[i]
#             print('    read',line,'then %s' % line[4],lines[i+2])
            if line[0] == '#': break  # end of discrete states
            L,Ex,J,par,cs = [float(x) for x in lines[i+2].split()]
            L = int(L); par = int(par)
#             excitedState = residual if L==0 else '%s_e%i' % (residual,L)
            excitedState = ejectile if L==0 else '%s%i' % (ejectile,L)
            previousX = discreteXsec.get(excitedState,None)
            if previousX is None: discreteXsec[excitedState] = []
            discreteXsec[excitedState].append([Ein,cs])
            print('    ',excitedState,'@',Ein,'=',cs*xs_scale)


os.system('mkdir ' + args.out )
for residual in discreteXsec.keys():
    foname = '%s/%s-to-%s' % (args.out,projectile,residual)
    print('Write',foname)
    fo = open(foname,'w')
    for e,cs in discreteXsec[residual]:
        print(e,cs*xs_scale, file=fo)
            
                
            
            
