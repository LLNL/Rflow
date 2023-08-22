#! /usr/bin/env python3

##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

from nuclear import *
import sys,os,math,argparse
from PoPs import database as databaseModule

from matplotlib import pyplot as plt
import numpy as np

lightnuclei = {'n':'N', 'H1':'P', 'H2':'D', 'H3':'T', 'He3':'HE3', 'He4':'A', 'photon':'G'}
quicknames = {'n':'n', 'H1':'p', 'H2':'d', 'H3':'t', 'He3':'h', 'He4':'a', 'photon':'g'}

lightA      = {'photon':0, 'n':1,  'H1':1, 'H2':2, 'H3':3, 'He3':3, 'He4':4}
lightZ      = {'photon':0, 'n':0,  'H1':1, 'H2':1, 'H3':1, 'He3':2, 'He4':2}
amu    = 931.494013

defaultPops = '../ripl2pops_to_Z20.xml'

def getmz(nucl):
    if nucl[0]=='2':
        multiplicity = 2 
        nucl = nucl[1:]
    else: 
        multiplicity = 1
        
    n = pops[nucl.replace('-','')]
    mass = n.getMass('amu')
    if nucl=='n' or nucl=='photon':
        charge = 0
    else:
        charge = n.nucleus.charge[0].value
    return (mass,charge)
    


parser = argparse.ArgumentParser(description='Prepare data for Rflow')

parser.add_argument('CN', type=str, help='compound-nucleus   e.g. N-15 or N15.')
parser.add_argument("-E", "--EnergyMax", type=float,  default="29.9", help="Maximum CN energy to allow. Default=19.9")
parser.add_argument("-e", "--EnergyMin", type=float,  default="0.1", help="Minimum CN energy to allow. Default=-1")
parser.add_argument("-L", "--LevelMax", type=int,  default="100", help="Maximum level to display")

parser.add_argument(      "--pops", type=str, default=defaultPops, help="pops files with all the level information from RIPL3. Default = %s" % defaultPops)
parser.add_argument("-t", "--tolerance", type=float,  default="0.1", help="Print warnings for level discrepancies larger than this")


args = parser.parse_args()
print('Command:',' '.join(sys.argv[:]) ,'\n')


pops = databaseModule.Database.read( args.pops )
CN = args.CN


if '-' in CN:
    name,Acn = CN.split('-')
else:
    name = CN[:2] if CN[1].isalpha() else CN[0]
    Acn = CN[len(name):]
    CN = '%s-%s' % (name,Acn)
CNgndsName = name+Acn
CNmass,CNcharge = getmz(CNgndsName)

Acn = int(Acn)
Zcn = elementsSymbolZ[name]

channels = []
Q = {}
partitions = []
for p in lightA.keys():
    Ap = lightA[p]
    Zp = lightZ[p]
    At = Acn - Ap
    Zt = Zcn - Zp
    if Ap<=At:
#         t = elementsZSymbolName[Zt][0] + '-%i' % At
        t = elementsZSymbolName[Zt][0] + '%i' % At
        channels.insert(0,(p,t))
print('\nBinary channels for',CN,':\n',channels)

for In in channels:
    p,t = In
    pn = lightnuclei[p]   # x4 name
    if pn == '2N': continue  # not a physical projectile
#     if pn == 'G': continue  # not a spectroscopic projectile
    pmass,pZ = getmz(p)
    tmass,tZ = getmz(t)
    if len(partitions) == 0:
        qvalue = 0.0
        Tmass = pmass + tmass

    Partmass = pmass + tmass
    qvalue = (Tmass - Partmass) * amu  # relative to the first partition
    CNthreshold = (Partmass - CNmass) * amu
    
    reaction =    p+'+'+t    
    print(reaction,' Q=',qvalue)
    Q[reaction] = qvalue
    partitions.append([reaction,CNthreshold])
    
sortedP = sorted(partitions, key = lambda v: v[1])  # ascending thresholds

print("\nSorted ",sortedP,'\n')

ipart = -1
for part in sortedP:
    reaction,threshold = part
    p,t = reaction.split('+')
    ipart += 1
    xs = np.asarray([ipart,ipart+0.75])
    
    ys = np.asarray([threshold,threshold])
    plt.plot(xs,ys, color="black", linestyle="dashed")
    pq = quicknames.get(p,p)
    plt.text(ipart,threshold-1,pq+'+'+t+'*')
    
    print('Reaction',reaction,'starting at %8.3f' % threshold)
    for lev in range(args.LevelMax):
        nuclide = '%s_e%i' % (t,lev) if lev > 0 else t
        try:
            state = pops[nuclide]
            Estar = state.energy[0].pqu('MeV').value
        except:
            if lev > 0: continue
            Estar = 0.0
        ECN = threshold + Estar
        if not (args.EnergyMin < ECN < args.EnergyMax): continue
        print('   level %i at %8.3f' % (lev,ECN))
        
        ys = np.asarray([ECN,ECN])

        plt.plot(xs,ys)

plt.savefig(open(CNgndsName+'levels.pdf','wb'),format='pdf')
plt.show(block=True)
