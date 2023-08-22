#! /usr/bin/env python3

##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

# <<BEGIN-copyright>>
# <<END-copyright>>

# This script takes an ENDF-like evaluation in an  gnd/XML file, 
# reads it in and rewrites it to a endf8*.data file for Rflow,
# with list if include projectiles in the title.
# Exclude excited states for now.

import sys
from fudge import reactionSuite

Emin = 0.1 # MeV
Emax = float(sys.argv[1])  # MeV
accRel = 0.01
ex2cm = 1.0

projReqd = sys.argv[2].split(',')
print('Ejectiles to save',projReqd)

for gndFile in sys.argv[3:] :
    dataFile = open(gndFile.replace('.xml','_'+sys.argv[2]+'.data'),'w')
    normFile = open(gndFile.replace('.xml','_'+sys.argv[2]+'.norms'),'w')
    
    gnd = reactionSuite.ReactionSuite.readXML_file( gndFile )
    gnd.convertUnits( {'eV':'MeV'} )
    p = gnd.projectile; t = gnd.target; incoming = p + ' + ' +t
    print(p,' : projectile defining the lab energy in first column', file=dataFile)
 
#     labels = gnd.reactions.labels
#     print('Reactions:',labels)
    for reactionList in  [gnd.reactions, gnd.sums.crossSections]:
        for reac in reactionList:
            if "_e" in reac.label: continue
            name = reac.label
    #         if True: # and reac.label != incoming and reac.label!='sumOfRemainingOutputChannels':
            rParts = name.split(' + ')
            ejectile = rParts[0]
            residual = ''
            if len(rParts)>=2: residual = rParts[1]
        
            print('\n',name,'cropped to [',Emin,Emax,'] giving',ejectile,residual, 'Exclude:',ejectile not in projReqd)
            if ejectile not in projReqd: continue
            name = name.replace(' + ','+')
            print(1,accRel, gndFile.replace('xml',name),1, accRel, gndFile, file=normFile)
            xsc = reac.crossSection[ 'eval' ]
            xsc = xsc.domainSlice(domainMin = Emin, domainMax = Emax )
            XYs =  xsc.toPointwise_withLinearXYs(lowerEps=1e-8)
            if ejectile == 'total':
                ejectile,residual = 'TOT',0
            group = 'endf8_'+p+','+ejectile

            for E,sig in XYs:
                sig = sig*1e3    # mb now
                if sig>0: print(E, -1, p,t,ejectile,residual, sig,sig*accRel, ex2cm, group, 'I', E, -1,  file=dataFile)
