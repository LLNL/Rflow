#! /usr/bin/env python3

##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################



from nuclear import *
import sys,os,math,argparse,numpy
from PoPs import database as databaseModule
from fudge import reactionSuite as reactionSuiteModule
from fudge import GNDS_file as GNDSTypeModule
from PoPs.chemicalElements.misc import *


from matplotlib import pyplot as plt
import numpy as np

plcolor = {0:"black", 1:"red", 2:"green", 3: "blue", 4:"yellow", 5:"brown", 14: "grey", 7:"violet",
            8:"cyan", 9:"magenta", 10:"orange", 11:"indigo", 12:"maroon", 13:"turquoise", 6:"darkgreen"}
pldashes = {0:'solid', 1:'dashed', 2:'dashdot', 3:'dotted'}
plsymbol = {0:".", 1:"o", 2:"s", 3: "D", 4:"^", 5:"<", 6: "v", 7:">",
            8:"P", 9:"x", 10:"*", 11:"p", 12:"D", 13:"P", 14:"X"}
            
amu    = 931.494013

parser = argparse.ArgumentParser(description='Plot CN levels')

parser.add_argument('Files', nargs='+', type=str, help='Input GNDS or PoPs files.')
parser.add_argument("-E", "--EnergyMax", type=float,  default="29.9", help="Maximum CN energy to allow. Default=19.9")
parser.add_argument("-e", "--EnergyMin", type=float,  default="0.1", help="Minimum CN energy to allow. Default=-1")
parser.add_argument("-L", "--LevelMax", type=int,  default="1000", help="Maximum level to display")
parser.add_argument("-N", "--Nucleus", type=str, help="GNDS name of CN if not already determined")


args = parser.parse_args()
print('Command:',' '.join(sys.argv[:]) ,'\n')

if args.Nucleus is not None: CN = args.Nucleus
nSrcs = len(args.Files)

stuffs = []
for file in  args.Files:
    type,info = GNDSTypeModule.type( file )
    name = info.get('name',None)
#     print(file,'has',name,':',type,info)
    
    if type=='PoPs':
        pops = databaseModule.database.readFile( file )
        stuffs.append(['pops',pops,file,name])
    
    if type=='reactionSuite':
        gnd=reactionSuiteModule.ReactionSuite.readXML_file( file )
        name = gnd.evaluation
        p,t = gnd.projectile,gnd.target
        PoPs = gnd.PoPs
        projectile = PoPs[p];
        target     = PoPs[t];
        if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
        if hasattr(target, 'nucleus'):     target = target.nucleus
        pMass = projectile.getMass('amu');   tMass = target.getMass('amu')
        CMass = pMass + tMass
        compoundA= int(CMass + 0.5)
        compoundZ = projectile.charge[0].value + target.charge[0].value
#         print('CN Z,A:',compoundZ,compoundA)
        CN = idFromZAndA(compoundZ,compoundA)
        lab2cm = tMass/CMass
        print(PoPs[CN])
        CNMass = PoPs[CN].getMass('amu')
        threshold = (CMass - CNMass) * amu
        
        stuffs.append(['rs',gnd,file,name,lab2cm,threshold])

print('Plot levels in ',CN)
nucleus = CN.lower() 

# print(stuffs)
spinsUsed = set()
ipart = 0
Emin = +1e3
Emax = -1e3
for stuff in stuffs:

    if stuff[0] == 'pops': 
        pops = stuff[1]
        file = stuff[2]
        name = stuff[3]
        
        print('Levels in PoPs file',file)
        levels = []
        for lev in range(args.LevelMax):
            nucleus = CN.lower()
            nuclide = '%s_e%i' % (nucleus,lev) if lev > 0 else nucleus
#             print('try',nuclide)
            try:
                state = pops[nuclide] #; print('state:',state)
                Estar = state.energy[0].pqu('MeV').value
            except:
                if lev > 0: continue
                Estar = 0.0
                
            J = state.spin[0].float('hbar')
            try:
                parity = state.parity[0].value
            except:
                parity = 0
            
            if not (args.EnergyMin < Estar < args.EnergyMax): continue
            levels.append([Estar,J,parity])
                
            if not (args.EnergyMin < Estar < args.EnergyMax): continue
#             print('   pops level %i at %8.3f' % (lev,Estar))    
    
    if stuff[0]=='rs':  # get resonance data from RMatrix part
        gnd = stuff[1]
        file = stuff[2]
        name = stuff[3]
        lab2cm = stuff[4]
        threshold = stuff[5]
        print('Levels in',name,' GNDS file',file,'starting at E %.3f' % threshold)
        
        rrr = gnd.resonances.resolved
        RMatrix = rrr.evaluated
        levels = []
        for Jpi in RMatrix.spinGroups:
            J = float(Jpi.spin)
            parity = int(Jpi.parity)
            R = Jpi.resonanceParameters.table
            rows = R.nRows
            E_poles =  R.getColumn('energy','MeV')    # lab MeV
#             print(E_poles)
            for E in E_poles:
                ECN = E * lab2cm + threshold
                if not (args.EnergyMin < ECN < args.EnergyMax): continue
                lev = len(levels)
                levels.append([ECN,J,parity])
#                 print('   gnds level %i at %8.3f in %.1f%s set from %.3f MeV lab' % (lev,ECN,J,parity,E))    
            
    sortedLevels = sorted(levels, key = lambda v: v[0]) 
#         for level in sortedLevels:
#             print(level)

    ipart += 1
    xs = np.asarray([ipart,ipart+0.75])
    ys = np.asarray([threshold,threshold])
    plt.plot(xs,ys, color="black", linestyle="dashed")

    ypos = max(threshold,args.EnergyMin)-1.0
    plt.text(ipart,ypos,name)
    
    for level in sortedLevels:
        ECN = level[0]
        J   = level[1]
        parity = level[2]
        styleDict = {1: 0, -1: 1, 0: 2}
        style = styleDict[parity]
        
        color = 14 if J < 0 else int(J)
        spinsUsed.add(J)
        Emin = min(Emin,ECN)
        Emax = max(Emax,ECN)
        ys = np.asarray([ECN,ECN])

        plt.plot(xs,ys, color=plcolor[color], linestyle=pldashes[style] )
    
sortedSpins = sorted(spinsUsed)
ipart += 1
print('Spins used:',spinsUsed)
Espacing = (Emax - Emin)/len(sortedSpins)
for J in  sorted(spinsUsed):
    xs = np.asarray([ipart,ipart+0.25])
    yJ = ypos + (J+1) * Espacing
    ys = np.asarray([yJ,yJ])
    color = 14 if J < 0 else int(J)
    
    plt.plot(xs,ys, color=plcolor[color] )
    Jname = '%s' % J if J >= 0 else '?'
    plt.text(ipart+0.10,yJ+Espacing*0.1,Jname)
    plt.xticks([])

plt.savefig(open('levels-%i-%s.pdf' % (nSrcs,CN),'wb'),format='pdf')
plt.show(block=True)
