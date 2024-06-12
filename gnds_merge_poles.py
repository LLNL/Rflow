#! /usr/bin/env python3

##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

# <<BEGIN-copyright>>
# <<END-copyright>>

import os,numpy
import argparse,sys

from fudge import GNDS_formatVersion as formatVersionModule
from PoPs import database as databaseModule
from fudge import GNDS_file as GNDSTypeModule
from pqu import PQU as PQUModule

extensionDefault = '.gm'

description1 = """Read two GNDS file into Fudge, add the poles from the second one into the first,
then write back to the GNDS/xml format with extension added.
"""

__doc__ = description1

parser = argparse.ArgumentParser( description1 )
parser.add_argument( 'input',                                                           help = 'GNDS file to merge into.' )
parser.add_argument( 'inputToAdd',                                                      help = 'GNDS file with data to add in.' )
parser.add_argument( '--energyUnit', type = str, default = None,                        help = 'Convert all energies in the gnds file to this unit.' )
parser.add_argument( 'output', nargs = '?', default = None,                             help = 'The name of the output file.' )
parser.add_argument( '-e', '--extension', default = extensionDefault,                   help = 'The file extension to add to the output file. Default = "%s"' % extensionDefault )

# parser.add_argument( '-s', '--scale', default = None, type=float,                       help = 'Scale all widths !' )
parser.add_argument( '-E', '--ETRIM', default = None, type=float,nargs=2,               help = 'Before merging: Remove all poles with energies between ETRIM[0] and ETRIM[1] for all spin groups')
parser.add_argument( '-t', '--trim', default = None, type=int, nargs=3,                 help = 'Before merging: Cut numbered trim[0]-trim[1] poles from spingroup trim[2].' )

parser.add_argument( '-p', '--path', default = None,                                    help = 'Path to write the file to. If absent, sent to same location as input.' )
parser.add_argument( '--skipCovariances', action = 'store_true',                        help = 'If present, any covariance files in are not written.' )
parser.add_argument( '--formatVersion', default = formatVersionModule.default, choices = formatVersionModule.allowed,
                                                                                        help = 'Specifies the GNDS format for the outputted file.  Default = "%s".' % formatVersionModule.default )

if( __name__ == '__main__' ) :

    args = parser.parse_args( )

    fileName = args.input
    fileName2 = args.inputToAdd

#     covariances = []
#     name, dummy = GNDSTypeModule.type( fileName )
#     if( name == databaseModule.database.moniker ) :
#         gnds = GNDSTypeModule.read( fileName )
#     else :
#         gnds = GNDSTypeModule.read( fileName )
#         if not args.skipCovariances:
#             try:
#                 if hasattr(gnds, 'loadCovariances'): covariances = gnds.loadCovariances()
#             except:
#                 print('WARNING: could not load covariance file(s).')

    gnds = GNDSTypeModule.read( fileName )
    gnds2= GNDSTypeModule.read( fileName2 )

    if( args.energyUnit is not None ) :
        gnds.convertUnits( { 'MeV' : args.energyUnit, 'eV' : args.energyUnit } )
        gnds2.convertUnits( { 'MeV' : args.energyUnit, 'eV' : args.energyUnit } )
#         for covarianceSuite in covariances:
#             covarianceSuite.convertUnits( { 'MeV' : args.energyUnit, 'eV' : args.energyUnit } )

    output = args.output
    path = args.path
    extension = args.extension

    PoPs = gnds.PoPs
    projectile = gnds.PoPs[gnds.projectile]
    target     = gnds.PoPs[gnds.target]
    elasticChannel = '%s + %s' % (gnds.projectile,gnds.target)
    if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
    if hasattr(target, 'nucleus'):     target = target.nucleus
    pZ = projectile.charge[0].value; tZ =  target.charge[0].value
    
    PoPs2 = gnds.PoPs
    projectile2 = gnds.PoPs[gnds.projectile]
    target2     = gnds.PoPs[gnds.target]
    elasticChannel2 = '%s + %s' % (gnds2.projectile,gnds2.target)
    if hasattr(projectile2, 'nucleus'): projectile2 = projectile2.nucleus
    if hasattr(target2, 'nucleus'):     target2 = target2.nucleus
    pZ2 = projectile2.charge[0].value; tZ2 =  target2.charge[0].value

    if(elasticChannel != elasticChannel2):
        print('Elastic channels different: not yet implemented')
        sys.exit()
    
    rrr = gnds.resonances.resolved
    Rm_Radius = gnds.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')
    bndx = RMatrix.boundaryCondition
    bndv = RMatrix.boundaryConditionValue
    IFG = RMatrix.reducedWidthAmplitudes
    
    rrr2 = gnds2.resonances.resolved
    Rm_Radius2 = gnds2.resonances.scatteringRadius
    Rm_global2 = Rm_Radius2.getValueAs('fm')
    RMatrix2 = rrr2.evaluated
    emin2 = PQUModule.PQU(rrr2.domainMin,rrr2.domainUnit).getValueAs('MeV')
    emax2 = PQUModule.PQU(rrr2.domainMax,rrr2.domainUnit).getValueAs('MeV')
    bndx2 = RMatrix2.boundaryCondition
    bndv2 = RMatrix2.boundaryConditionValue
    IFG2 = RMatrix2.reducedWidthAmplitudes
    

    channels = []
    pair = 0
    inpair = None
    chargedElastic = False
    ReichMoore = False
    print('\nChannels:')
    for partition in RMatrix.resonanceReactions:
        kp = partition.label
        channels.append(kp)
        pair += 1
        print('  reaction "%s"' % kp,' (eliminated)' if partition.eliminated else '')
        
    print('\nPoles:')
    jset = 0
    spinGroups = []
    pwSets = {}
    for Jpi in RMatrix.spinGroups:
        R = Jpi.resonanceParameters.table
        rows = R.nRows
        cols = R.nColumns
        parity = '+' if int(Jpi.parity) > 0 else '-'
        spinGroup = '%5.1f%s' % (Jpi.spin,parity)
        print('  J,pi =%5.1f%s, channels %3i, poles %3i : #%i' % (Jpi.spin,parity,cols,rows,jset) )
        spinGroups.append(spinGroup)
#         print(R.data)
        E_poles = R.data   # lab MeV
        for n in range(rows):
            E = R.data[n][0]
            widths = []
            for c in range(cols-1):
               widths.append(R.data[n][1+c])

        pwLists = []
        pwColumns= []
        for ch in Jpi.channels:
            n = ch.columnIndex
            L = ch.L
            S = ch.channelSpin
            rr = ch.resonanceReaction
            pw = '%s,%s,%s' % (rr,L,S)
            pwLists.append(pw)
            pwColumns.append(n)
          
        pwSets[spinGroup] = [pwLists,pwColumns]
        print('   Partial waves:',pwSets[spinGroup])
        if args.ETRIM is not None:
            emin, emax = args.ETRIM 
            extension = '.E%s-%s' % (emin, emax ) 
            print('   Cut poles %s < E < %s from Jpi=%5.1f%s' % (emin, emax ,Jpi.spin,parity) )
            toCut = [] 
            for n in range(rows):
                E = R.data[n][0]
                if emin < E < emax: 
                    toCut.append(n)
            orderedCuts = sorted(toCut)
            for n in reversed(orderedCuts):
                print('     Cut E=',R.data[n][0])
                del R.data[n]
            print('     ',R.nRows,'rows now')
        if args.trim is not None:
            first, last, nset = args.trim 
            extension = '.t%i-%i' % (first,last ) 
            if jset==nset:
                print('   Cut poles %i-%i from Jpi=%5.1f%s' % (first,last,Jpi.spin,parity) )
                for n in range(last,first-1,-1):
                    print('     Cut E=',R.data[n][0])
                    del R.data[n]
                print('     ',R.nRows,'rows now\n:')
        jset += 1
        
    print('\nChannels2:')
    channels2 = []
    channel_in_merged = []
    pair2 = 0
    inpair2 = None
    chargedElastic2 = False
    ReichMoore2 = False
    for partition in RMatrix2.resonanceReactions:
        kp2 = partition.label
        channels2.append(kp2)
        try:
            cim = channels.index(kp2)
        except:
            print('      Incoming channel',kp2,'not found in target list',channels,'Ignored')
            cim = None
        channel_in_merged.append(cim)
        pair2 += 1
        print('  reaction2 "%s"' % kp2,'to',cim)
                
    print('channel_in_merged:',channel_in_merged,'\n')

    spinGroup_in_merged = []
    jset2 = 0
    for Jpi2 in RMatrix2.spinGroups:
        R = Jpi2.resonanceParameters.table
        rows = R.nRows
        cols = R.nColumns
        parity = '+' if int(Jpi2.parity) > 0 else '-'
        print('  J,pi =%5.1f%s, channels %3i, poles %3i : #%i' % (Jpi2.spin,parity,cols,rows,jset2) )
        spinGroup2 = '%5.1f%s' % (Jpi2.spin,parity)
        try:
            sim = spinGroups.index(spinGroup2)
        except:
#             print('Incoming channel',spinGroup2,'not found in merged list',spinGroups,'Ignored')
            sim = None
        spinGroup_in_merged.append(sim)
        jset2 += 1
    print('spinGroup_in_merged:',spinGroup_in_merged,'\n')
       
          
    jset2 = 0
    for Jpi2 in RMatrix2.spinGroups:
        R2 = Jpi2.resonanceParameters.table
        rows2 = R2.nRows
        cols2 = R2.nColumns
        parity2 = '+' if int(Jpi2.parity) > 0 else '-'
        print('  J,pi =%5.1f%s, channels %3i, poles %3i : #%i' % (Jpi2.spin,parity2,cols2,rows2,jset2) )
        spinGroup2 = '%5.1f%s' % (Jpi2.spin,parity2)
        E_poles = R2.data   # lab MeV
        for n in range(rows2):
            E = R2.data[n][0]
            widths = []
            for c in range(cols2-1):
               widths.append(R2.data[n][1+c])
            print('    E= %8.5f,' % E,widths,' #',n)
        
        jset = spinGroup_in_merged[jset2]
        if jset is None:
            print("This set",jset," does not exist in target collection")
            continue
        Jpi = RMatrix.spinGroups[jset]
        print('    Put in merged set',jset,':',Jpi.spin,Jpi.parity)
        R = Jpi.resonanceParameters.table
        cols = R.nColumns
        
        pwLists,pwColumns = pwSets[spinGroup2]
        newCol = [None for i in range(cols2)]
        for ch in Jpi2.channels:
            ci2 = ch.columnIndex
            L = ch.L
            S = ch.channelSpin
            rr = ch.resonanceReaction
            pw = '%s,%s,%s' % (rr,L,S) 
            try:
                targetpw = pwLists.index(pw)
            except:
#                 print("This partial wave",pw," does not exist in target collection",pwLists)
                print("     Partial wave",pw," not merged")
                continue
            targetCol = pwColumns[targetpw]
            newCol[ci2] = targetCol
            
        for n2 in range(rows2):
            newRow = [0.0 for i in range(cols)]
            newRow[0] = R2.data[n2][0]  # energy
            for ci2 in range(cols2-1):
                if newCol[ci2] is None: continue
                newRow[newCol[ci2]] = R2.data[n2][ci2]
            nTo = R.nRows
            for n in range(R.nRows):
#                 print('If',R[n][0], ' > ', newRow[0],'then use',n,':', R[n][0] > newRow[0])
                if newRow[0] < R[n][0]:
                    nTo = n
                    break
#             print('     Insert at',nTo,'from',n2,':',newRow) 
            R.data.insert(nTo,newRow)
        
        print('    New energies:',[R.data[n][0] for n in range(R.nRows)])
        jset2 += 1
        

    if( output is None ) :
        if( path is None ) : path = os.path.dirname( fileName )
        output = os.path.join( path, os.path.basename( fileName ) ) + extension

    gnds.saveToFile( output, formatVersion = args.formatVersion )

