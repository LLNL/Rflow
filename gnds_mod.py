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
import argparse

from fudge import GNDS_formatVersion as formatVersionModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule
from PoPs import database as databaseModule
from fudge import GNDS_file as GNDSTypeModule
from pqu import PQU as PQUModule
import xData.constant as constantModule
import xData.axes as axesModule

extensionDefault = '.gm'

description1 = "Read a GNDS file into Fudge, then write back to the GNDS/xml format with extension added."

__doc__ = description1

parser = argparse.ArgumentParser( description1 )
parser.add_argument( 'input',                                                           help = 'GNDS and/or PoPs file to translate.' )
parser.add_argument( 'output', nargs = '?', default = None,                             help = 'The name of the output file.' )
parser.add_argument( '--energyUnit', type = str, default = None,                        help = 'Convert all energies in the gnds file to this unit.' )
parser.add_argument( '-e', '--extension', default = extensionDefault,                   help = 'The file extension to add to the output file. Default = "%s"' % extensionDefault )
parser.add_argument( '-p', '--path', default = None,                                    help = 'Path to write the file to. If absent, sent to same location as input.' )
parser.add_argument( '-s', '--scale', default = None, type=float,                       help = 'Scale all widths.' )
parser.add_argument( '-z', '--zeros', default = None, type=float,                       help = 'Change zeros to this value.' )
parser.add_argument( '-E', '--ETRIM', default = None, type=float,nargs=2,               help = 'Remove all poles with energies between ETRIM[0] and ETRIM[1] for all spin groups')
parser.add_argument( '-t', '--trim', default = None, type=int, nargs=3,                 help = 'Cut numbered trim[0]-trim[1] poles from spingroup trim[2].' )
parser.add_argument( '-r', '--radii', default = None, type=float, nargs='+',              help = 'Reset channel radii to this sequence of projectiles.' )

parser.add_argument( '--skipCovariances', action = 'store_true',                        help = 'If present, any covariance files in are not written.' )
parser.add_argument( '--formatVersion', default = formatVersionModule.default, choices = formatVersionModule.allowed,
                                                                                        help = 'Specifies the GNDS format for the outputted file.  Default = "%s".' % formatVersionModule.default )

if( __name__ == '__main__' ) :

    args = parser.parse_args( )

    fileName = args.input
    gnds = GNDSTypeModule.read( fileName )

    covariances = []
#   name, dummy = GNDSTypeModule.type( fileName )
#   if( name == databaseModule.database.moniker ) :
#       gnds = GNDSTypeModule.read( fileName )
#   else :
#       gnds = GNDSTypeModule.read( fileName )
#       if not args.skipCovariances:
#           try:
#               if hasattr(gnds, 'loadCovariances'): covariances = gnds.loadCovariances()
#           except:
#               print('WARNING: could not load covariance file(s).')

    if( args.energyUnit is not None ) :
        gnds.convertUnits( { 'MeV' : args.energyUnit, 'eV' : args.energyUnit } )
        for covarianceSuite in covariances:
            covarianceSuite.convertUnits( { 'MeV' : args.energyUnit, 'eV' : args.energyUnit } )

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
    identicalParticles = gnds.projectile == gnds.target
#     rStyle = fitStyle.label
    
    rrr = gnds.resonances.resolved
    Rm_Radius = gnds.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')
    bndx = RMatrix.boundaryCondition
    bndv = RMatrix.boundaryConditionValue
    IFG = RMatrix.reducedWidthAmplitudes
    
    if args.radii is not None:
        newRadii = args.radii
        changedRadii = len(newRadii)

    channels = {}
    pair = 0
    inpair = None
    ReichMoore = False
    damping = 0
    print('\nChannels:')
    Rm_Radius = gnds.resonances.scatteringRadius
    
    Ejectiles = []
    for partition in RMatrix.resonanceReactions:
        kp = partition.label
        if partition.eliminated: # no two-body kinematics
            partitions[kp] = None
            ReichMoore = True
            damping = 1
            continue
        channels[pair] = kp
        prmax = Rm_Radius
        if partition.scatteringRadius is not None:
            prmax =  partition.getScatteringRadius().getValueAs('fm')

        changed = ''
        if args.radii is not None:
            ejectile = partition.ejectile
            if ejectile not in Ejectiles:
                Ejectiles.append(ejectile)
            index = Ejectiles.index(ejectile)
            if index < changedRadii: 
                newR = newRadii[index]
                newRadius = scatteringRadiusModule.ScatteringRadius(
                        constantModule.Constant1d(newR, domainMin=emin, domainMax=emax,
                            axes=axesModule.Axes( labelsUnits={1: ('energy_in', 'MeV'), 0: ('radius', 'fm')})) )
                partition.scatteringRadius = newRadius
                changed = ' changed to %s' % newR

        pair += 1
        print('  reaction "%s"' % kp,' (eliminated)' if partition.eliminated else '','R =',prmax,changed)
        
    print('\nPoles:')
    jset = 0
    for Jpi in RMatrix.spinGroups:
        R = Jpi.resonanceParameters.table
        rows = R.nRows
        cols = R.nColumns
        parity = '+' if int(Jpi.parity) > 0 else '-'
        print('  J,pi =%5.1f%s, channels %3i, poles %3i : #%i' % (Jpi.spin,parity,cols,rows,jset) )
#         print(R.data)
        E_poles = R.data   # lab MeV
        for n in range(rows):
            E = R.data[n][0]
            D = R.data[n][damping] if damping==1 else 0
            widths = []
            for c in range(cols-1-damping):
               widths.append(R.data[n][1+damping+c])
            print('    E= %8.5f, D = %4.1e,' % (E,D),widths,' #',n)
            if args.scale is not None:
                newwidths = []
                for c in range(cols-1-damping):
                   newwidths.append(widths[c] * args.scale )              
                   R.data[n][1+damping+c] = newwidths[c]                
                print('    New: E= %8.5f, D = %4.1e,' % (E,D),newwidths)
                extension = '.s%s' % args.scale 
            if args.zeros is not None:
                newwidths = []
                for c in range(cols-1-damping):
                   newwidths.append(widths[c] if abs(widths[c])>1e-20 else  args.zeros )              
                   R.data[n][1+damping+c] = newwidths[c]                
                print('    New: E= %8.5f, D = %4.1e,' % (E,D),newwidths)
                extension = '.z%s' % args.zeros 
        if args.ETRIM is not None:
            emin, emax = args.ETRIM 
            extension = '.E%s-%s' % (emin, emax ) 
            print('Cut poles %s < E < %s from Jpi=%5.1f%s' % (emin, emax ,Jpi.spin,parity) )
            toCut = [] 
            for n in range(rows):
                E = R.data[n][0]
                if emin < E < emax: 
                    toCut.append(n)
            orderedCuts = sorted(toCut)
            for n in reversed(orderedCuts):
                print('   Cut E=',R.data[n][0])
                del R.data[n]
            print(R.nRows,'rows now')
        if args.trim is not None:
            first, last, nset = args.trim 
            extension = '.t%i-%i' % (first,last ) 
            if jset==nset:
                print('Cut poles %i-%i from Jpi=%5.1f%s' % (first,last,Jpi.spin,parity) )
                for n in range(last,first-1,-1):
                    print('   Cut E=',R.data[n][0])
                    del R.data[n]
                print(R.nRows,'rows now')
        jset += 1


    if( output is None ) :
        if( path is None ) : path = os.path.dirname( fileName )
        output = os.path.join( path, os.path.basename( fileName ) ) + extension

    gnds.saveToFile( output, formatVersion = args.formatVersion )

    for covarianceSuite in covariances:
        output = os.path.join( os.path.dirname( output ), os.path.basename( covarianceSuite.sourcePath ) ) + extension
        covarianceSuite.saveToFile( output, formatVersion = args.formatVersion )
