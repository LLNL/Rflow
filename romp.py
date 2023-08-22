#!/usr/bin/env python3

##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

import times
tim = times.times()

import os,math,numpy,cmath,pwd,sys,time,json
from gomp import Gomp


from fudge import reactionSuite as reactionSuiteModule
from fudge import styles        as stylesModule
from pqu import PQU as PQUModule

REAL = numpy.double
CMPLX = numpy.complex128
INT = numpy.int32
realSize = 8  # bytes
pi = 3.1415926536


############################################## main

if __name__=='__main__':
    import argparse,re

    print('\nRomp')
    cmd = ' '.join([t if '*' not in t else ("'%s'" % t) for t in sys.argv[:]])
    print('Command:',cmd ,'\n')


    # Process command line options
    parser = argparse.ArgumentParser(description='Compare R-matrix Cross sections with Data')
    parser.add_argument('inFile', type=str, help='The  intial gnds R-matrix set' )
    parser.add_argument("-O", "--OmpFile", type=str, help='Optical model parameters to use' )
    parser.add_argument("-M", "--Model", type=str, default='B', help="Model to link |S|^2 and widths. A: log; B: lin; X")
    parser.add_argument("-D", "--Dspacing", type=float,  help="Energy spacing of optical poles")
    parser.add_argument("-L", "--LevelDensity", type=str,  help="Level-density parameter file for compound nucleus")
    parser.add_argument("-P", "--PorterThomas", type=int, default=0, help="rwa: 0: positive, <0: Porter-Thomas, >0: random sign")
    parser.add_argument("-R", "--Rmax", type=float, default = 20.0,  help="Radius limit for optical potentials.")
    parser.add_argument("-F", "--FormalWidths", action="store_true", help="Optical model widths taken as Formal, not Observed. Default Observed (previously Formal)")
    
    parser.add_argument("-e", "--emin", type=float, default = 0.5,  help="Min cm energy for optical poles.")
    parser.add_argument("-E", "--EMAX", type=float, default = 20, help="Max cm energy for optical poles")
    parser.add_argument("-j", "--jmin", type=float, default = 0, help="Max CN spin for optical poles")
    parser.add_argument("-J", "--JMAX", type=float, default = 8, help="Max CN spin for optical poles")
    parser.add_argument("-Y", "--YRAST", type=float, default = 0.3,  help="Min CN energy = max(emin , YRAST*J*(J+1) )")
    parser.add_argument("-H", "--Hcm"  , type=float, default = 0.1, help="Radial step size")
    parser.add_argument("-o", "--offset"  , type=float, default = 0., help="Shift new poles by (J + pi/2)* offset")
    
    parser.add_argument("-C", "--Convolute", type=float,  help="Calculate MLBW excitations xsecs. Value = Width of gaussian  smoothing, 0 for no smoothing.")
    parser.add_argument("-S", "--Stride", type=int, default=5, help="Stride for accessing non-uniform grid template")
    
    parser.add_argument("-s", "--single", action="store_true", help="Single precision: float32, complex64")
    parser.add_argument("-t", "--tag", type=str, default='', help="Tag identifier for this run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debugging output (more than verbose)")
    args = parser.parse_args()

    if args.single:
        REAL = numpy.float32
        CMPLX = numpy.complex64
        INT = numpy.int32
        realSize = 4  # bytes
    ComputerPrecisions = (REAL, CMPLX, INT, realSize)

    gnds=reactionSuiteModule.ReactionSuite.readXML_file(args.inFile)
    p,t = gnds.projectile,gnds.target        
    rrr = gnds.resonances.resolved
    eminG = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emaxG = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')
    eminG = min(eminG, args.emin)
    emaxG = args.EMAX # max(emaxG, args.EMAX)
    emin = args.emin
    emax = emaxG
    
# parameter input for computer method
    base = args.inFile
    if args.single:           base += 's'
    
    NewLevels = args.Dspacing is not None or args.LevelDensity is not None
    if  NewLevels:
        if args.Dspacing is not None:
            print(' Make optical poles spaced by',args.Dspacing,'in range [',emin,',',emax,'] in lab MeV for projectile',p)
        if args.LevelDensity is not None:
            print(' Make optical poles spaced by parameters ',args.LevelDensity,'in range [',emin,',',emax,'] in lab MeV for projectile',p)
            ld = open(args.LevelDensity,'r')
            LDparameters = ld.readlines()
            LevelParms = {}
            LevelParms['low_e_mod']  = 1.0 # 9
            for line in LDparameters:
                parts = line.strip().split('=')
                if len(parts) == 2 and 'UNAVAILABLE' not in parts[1] and 'Rho' not in parts[0]:
                      parts[1] = parts[1].strip().replace('for positive and negative parities','')
                      LevelParms[parts[0].strip()] = float(parts[1]) if 'G & C' not in parts[1] else 'G & C'
            print('Level Density Parameters:\n',LevelParms)

        print(' using optical potentials from',args.OmpFile,' and model ',args.Model,'to map |S|^2 to widths.\n\n')
#       rrr.domainMax = args.EMAX
    
    
        f = open( args.OmpFile )
        omp_lines = f.readlines( )
        f.close( )    
        optical_potentials = {}
        for l in omp_lines:
            parts = l.split()
            proj = parts[0]
            targ = parts[1]
            pair = '%s + %s' % (proj,targ)
            om_parameters = [float(v) for v in parts[2:]]
            optical_potentials[pair] = om_parameters

# data input
        base += '+%s' % args.OmpFile.replace('.omp','')

        if args.Model      is not None: base += '-%s' % args.Model
        if not args.FormalWidths : base += 'O'
        if args.emin       is not None: base += '-e%s' % args.emin
        if args.EMAX       is not None: base += '-E%s' % args.EMAX
        if args.jmin    > 0.0: base += '-j%s' % args.jmin
        if args.JMAX    != 8.0        : base += '-J%s' % args.JMAX
        if args.Rmax    != 20.        : base += '-R%s' % args.Rmax
    
        if args.offset  > 0.0: base += '-o%s' % args.offset
        if args.Dspacing       is not None: base += '-D%s' % args.Dspacing
        if args.LevelDensity   is not None: base += '-L%s' % args.LevelDensity.replace('.dat','')
        if args.PorterThomas > 0: base += ':P%s' % args.PorterThomas
        if args.PorterThomas < 0: base += ':Pm%s' % abs(args.PorterThomas)
        if args.YRAST    > 0.0: base += '-Y%s' % args.YRAST

    else:
        optical_potentials = None

    if args.Convolute       is not None: base += '-C%s' % args.Convolute
    if args.tag != '': base = base + '_'+args.tag

#     print("        finish setup: ",tim.toString( ))
 
    Gomp(gnds,base,emin,emax,args.jmin,args.JMAX,args.Dspacing,LevelParms,args.PorterThomas,optical_potentials,
         args.FormalWidths,args.Rmax,args.Model,args.YRAST,args.Hcm,args.offset,args.Convolute,args.Stride,
         args.verbose,args.debug,args.inFile,ComputerPrecisions,tim)
    
    if NewLevels:
        newFitFile = base  + '-opt.xml'
        open( newFitFile, mode='w' ).writelines( gnds.toXML( ) )
        print('Written new gnds file:',newFitFile)
    

#     print("Final Romp: ",tim.toString( ))
    print("Target stdout:",base + '.out')
