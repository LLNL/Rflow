#!/usr/bin/env python3

import times
tim = times.times()

import os,math,numpy,cmath,pwd,sys,time,json
from gomp import Gomp


from fudge.gnds import reactionSuite as reactionSuiteModule
from fudge.gnds import styles        as stylesModule
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
    parser.add_argument('ompFile', type=str, help='Optical model parameters to use' )

    parser.add_argument("-M", "--Model", type=str, default='B', help="Model to link |S|^2 and widths. A: log; B: lin")
    parser.add_argument("-D", "--Dspacing", type=float,default = 1,   help="Energy spacing of optical poles")
    
    parser.add_argument("-e", "--emin", type=float, default = 11,  help="Min cm energy for optical poles.")
    parser.add_argument("-E", "--EMAX", type=float, default = 20, help="Max cm energy for optical poles")
    parser.add_argument("-j", "--jmin", type=float, default = 0, help="Max CN spin for optical poles")
    parser.add_argument("-J", "--JMAX", type=float, default = 5, help="Max CN spin for optical poles")
    parser.add_argument("-Y", "--YRAST", type=float, default = 1,  help="Max CN spin(E) = max(jmin + YRAST*sqrt(E), JMAX)")
    parser.add_argument("-H", "--Hcm"  , type=float, default = 0.1, help="Radial step size")
    parser.add_argument("-o", "--offset"  , type=float, default = 0., help="Shift new poles by (J + pi/2)* offset")
    

    parser.add_argument("-s", "--single", action="store_true", help="Single precision: float32, complex64")
    parser.add_argument("-t", "--tag", type=str, default='', help="Tag identifier for this run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-g", "--debug", action="store_true", help="Debugging output (more than verbose)")
    args = parser.parse_args()

    if args.single:
        REAL = numpy.float32
        CMPLX = numpy.complex64
        INT = numpy.int32
        realSize = 4  # bytes
    ComputerPrecisions = (REAL, CMPLX, INT, realSize)

    gnds=reactionSuiteModule.readXML(args.inFile)
    p,t = gnds.projectile,gnds.target        
    rrr = gnds.resonances.resolved
    eminG = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emaxG = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')
    eminG = min(eminG, args.emin)
    emaxG = args.EMAX # max(emaxG, args.EMAX)
    emin = args.emin
    emax = emaxG
    print(' Make optical poles spaced by',args.Dspacing,'in range [',emin,',',emax,'] in lab MeV for projectile',p)
    print(' using optical potentials from',args.ompFile,' and model ',args.Model,'to map |S|^2 to widths.\n\n')
    rrr.domainMax = args.EMAX
    
    
    f = open( args.ompFile )
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
    
    
# parameter input for computer method
    base = args.inFile
    if args.single:           base += 's'
# data input
    base += '+%s' % args.ompFile.replace('.omp','')

    if args.Model      is not None: base += '-%s' % args.Model
    if args.emin       is not None: base += '-e%s' % args.emin
    if args.EMAX       is not None: base += '-E%s' % args.EMAX
    if args.jmin    > 0.0: base += '-j%s' % args.jmin
    if args.JMAX       is not None: base += '-J%s' % args.JMAX
    
    if args.offset  > 0.0: base += '-o%s' % args.offset
    if args.Dspacing       is not None: base += '-d%s' % args.Dspacing

    if args.tag != '': base = base + '_'+args.tag

#     print("        finish setup: ",tim.toString( ))
 
    Gomp(gnds,base,emin,emax,args.jmin,args.JMAX,args.Dspacing,optical_potentials,args.Model,args.Hcm,args.offset,  
         args.verbose,args.debug,args.inFile,ComputerPrecisions,tim)
    
    newFitFile = base  + '-opt.xml'
    open( newFitFile, mode='w' ).writelines( line+'\n' for line in gnds.toXMLList( ) )
    print('Written new fit file:',newFitFile)
    

#     print("Final Romp: ",tim.toString( ))
    print("Target stdout:",base + '.out')
