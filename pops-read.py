#! /usr/bin/env python3

##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

from PoPs import database
import sys

print('Read',sys.argv[1])
pops = database.Database.read(sys.argv[1])
for p in sys.argv[2:]:
    print('Read',p)
    pops.addFile(p)

# with open('pops-endf8-global.xml','w') as f:
#     print(pops.toXML(), file=f)
print(pops.keys())

maxLevel = {}
for nuclide in pops.keys():
    if '_e' not in nuclide or nuclide[0].islower(): continue
#     print (nuclide)
    element,level = nuclide.split('_e')
    level = int(level)
    
    if element in maxLevel.keys():
        maxLevel[element] = max(maxLevel[element], level)
    else:
        maxLevel[element] = level
        
        
for element in maxLevel.keys():
    print('%s max level  %3i' % (element,maxLevel[element]))


