#! /usr/bin/env python3

##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

import sys

npts = int(sys.argv[1])

for infile in sys.argv[2:]:

    onfile = infile + 'dl'
    print('Convert %s to %s' % (infile,onfile))
    try:
        inf = open(infile,'r')
    except:
        continue
        print("   read failed")
    onf = open(onfile,'w')
    mlast = 1e10
    others = ''
    c = 0
    for l in inf.readlines():
        if '#' in l: continue
        parts = l.split()
        m = float(parts[0])
        others = parts[1:]
        if m < mlast:
            mp = max(m,1e-50)
            print(c,mp/npts,' '.join(others),file=onf)
            mlast = m
        c += 1
