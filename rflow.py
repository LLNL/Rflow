#!/usr/bin/env python3

import times
tim = times.times()

import os,math,numpy,cmath,pwd,sys,time,json

from wrapup import plotOut,saveNorms2gnds
# from gflowe import Gflow
from gflow import Gflow
from write_covariances import write_gnds_covariances
from printExcitationFunctions import *

from fudge.gnds import reactionSuite as reactionSuiteModule
from fudge.gnds import styles        as stylesModule
from pqu import PQU as PQUModule

REAL = numpy.double
CMPLX = numpy.complex128
INT = numpy.int32
realSize = 8  # bytes
pi = 3.1415926536

# print("First imports done rflow: ",tim.toString( ))


# TO DO:
#   Reich-Moore widths to imag part of E_pole like reconstructxs_TF.py
#   Multiple GPU strategies
#   Estimate initial Hessian by 1+delta parameter shift. Try various delta to make BFGS search smoother
#   Options to set parameter and search again.
#   Reread snap file (eg if crash) for re-initializing same search 

# Search options:
#   Fix or search on Reich-Moore widths
#   Command input, e.g. as with Sfresco?

# Maybe:
#   Fit specific Legendre orders

# Doing:



############################################## main

if __name__=='__main__':
    import argparse,re

    print('\nRflow')
    cmd = ' '.join([t if '*' not in t else ("'%s'" % t) for t in sys.argv[:]])
    print('Command:',cmd ,'\n')

    Gdefault = 0.001

    # Process command line options
    parser = argparse.ArgumentParser(description='Compare R-matrix Cross sections with Data')
    parser.add_argument('inFile', type=str, help='The  intial gnds R-matrix set' )
    parser.add_argument('dataFile', type=str, help='Experimental data to fit' )
    parser.add_argument('normFile', type=str, help='Experimental norms for fitting' )
    parser.add_argument("-x", "--exclude", metavar="EXCL", nargs="*", help="Substrings to exclude if any string within group name")

    parser.add_argument("-1", "--norm1", action="store_true", help="Start with all norms=1")
    parser.add_argument("-F", "--Fixed", type=str, nargs="*", help="Names of variables (as regex) to keep fixed in searches")
    parser.add_argument("-n", "--normsfixed", action="store_true",  help="Fix all physical experimental norms (but not free norms)")

    parser.add_argument("-r", "--restarts", type=int, default=0, help="max restarts for search")
    parser.add_argument("-B", "--Background", type=float, default="25",  help="Pole energy (lab) above which are all distant poles. Fixed in  searches.")
    parser.add_argument(      "--BG", action="store_true",  help="Include BG in name of background poles")
    parser.add_argument("-R", "--ReichMoore", action="store_true", help="Include Reich-Moore damping widths in search")
    parser.add_argument(      "--LMatrix", action="store_true", help="Use level matrix method if not already Brune basis")
    parser.add_argument(      "--groupAngles", type=int, default="1",  help="Unused. Number of energy batches for T2B transforms, aka batches")
    parser.add_argument("-a", "--anglesData", type=int, help="Max number of angular data points to use (to make smaller search). Pos: random selection. Neg: first block")
    parser.add_argument("-m", "--maxData", type=int, help="Max number of data points to use (to make smaller search). Pos: random selection. Neg: first block")
    parser.add_argument("-e", "--emin", type=float, help="Min cm energy for gnds projectile.")
    parser.add_argument("-E", "--EMAX", type=float, help="Max cm energy for gnds projectile.")
    parser.add_argument("-p", "--pmin", type=float, help="Min energy of R-matrix pole to fit, in gnds cm energy frame. Overrides --Fixed.")
    parser.add_argument("-P", "--PMAX", type=float, help="Max energy of R-matrix pole to fit. If p>P, create gap.")
    parser.add_argument("-d", "--dmin", type=float, help="Min energy of R-matrix pole to fit damping, in gnds cm energy frame.")
    parser.add_argument("-D", "--DMAX", type=float, help="Max energy of R-matrix pole to fit damping. If d>D, create gap.")
    parser.add_argument("-L", "--Lambda", type=float, help="Use (E-dmin)^Lambda to modulate all damping widths at gnds-scattering cm energy E.")
    parser.add_argument(      "--ABES", action="store_true", help="Allow Brune Energy Shifts.  Use inexact method")
    parser.add_argument("-G", "--Grid", type=float, default=Gdefault, help="Make energy grid with this energy spacing (MeV) for 1d interpolation, default %s" % Gdefault)

    parser.add_argument("-S", "--Search", type=str, help="Search minimization target.")
    parser.add_argument("-I", "--Iterations", type=int, default=2000, help="max_iterations for search")
    parser.add_argument("-i", "--init",type=str, nargs="*", help="iterations and snap file name for starting parameters")
    parser.add_argument("-w", "--widthWeight", type=float, default=0.0, help="Add widthWeight*vary_widths**2 to chisq during searches")
    parser.add_argument("-X", "--XCLUDE", type=float,  help="Make dataset*3 with data chi < X (e.g. X=3). Needs -C data.")
    
    parser.add_argument(      "--Large", type=float, default="40",  help="'large' threshold for parameter progress plotts.")
    parser.add_argument("-C", "--Cross_Sections", action="store_true", help="Output fit and data files, for json and grace")
    parser.add_argument("-c", "--compound", action="store_true", help="Plot -M and -C energies on scale of E* of compound system")
    parser.add_argument("-T", "--TransitionMatrix",  type=int, default=1, help="Produce cross-section transition matrix functions in *tot_a and *fch_a-to-b")

    parser.add_argument("-s", "--single", action="store_true", help="Single precision: float32, complex64")
    parser.add_argument("-M",  "--Multi", type=int, default=0, help="Which Mirrored Strategy in TF")

    parser.add_argument(      "--datasize", type=float,  metavar="size", default="0.2", help="Font size for experiment symbols. Default=0.2")
    parser.add_argument("-l", "--logs", type=str, default='', help="none, x, y or xy for plots")
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

    gnd=reactionSuiteModule.readXML(args.inFile)
    p,t = gnd.projectile,gnd.target
    PoPs = gnd.PoPs
    projectile = PoPs[p];
    target     = PoPs[t];
    pMass = projectile.getMass('amu');   tMass = target.getMass('amu')
    lab2cmi = tMass / (pMass + tMass)
        
    rrr = gnd.resonances.resolved
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')
    if args.emin is not None: emin = args.emin
    if args.EMAX is not None: emax = args.EMAX
    print(' Trim incoming data within [',emin,',',emax,'] in lab MeV for projectile',p)
            
# Previous fitted norms:
# any variable or data namelists in the documentation?
    docVars = []
    docData = []
    RMatrix = gnd.resonances.resolved.evaluated    
    try:
        computerCodeFit = RMatrix.documentation.computerCodes['R-matrix fit']
        ddoc    = computerCodeFit.inputDecks[-1]
        for line in ddoc.body.split('\n'):
            if '&variable' in line.lower() :  docVars += [line]
            if '&data'     in line.lower() :  docData += [line]
        previousFit = True
    except:
        computerCodeFit = None
        previousFit = False
        
    Fitted_norm = {}
    for line in docVars:
        if 'kind=5' in line:
            name = line.split("'")[1].strip()
            datanorm = float(line.split('datanorm=')[1].split()[0])
            Fitted_norm[name] = datanorm
            if args.debug: print("Previous norm for %-20s is %10.5f" % (name,datanorm) )

    pair = 0
    partitions = {}
    pins = []
    for partition in RMatrix.resonanceReactions:
        kp = partition.label
        if partition.eliminated: continue
        p,t = partition.ejectile,partition.residual
        partitions[kp] = pair
        pins.append(kp.replace(' ',''))
        pair += 1
#                Ecm = E/cm2lab[ipair] + QI[ipair]


    f = open( args.dataFile )
    projectile4LabEnergies =f.readline().split()[0]
    lab2cmd = None
    for partition in RMatrix.resonanceReactions:
        reaction = partition.reactionLink.link
        p,t = partition.ejectile,partition.residual
        if partition.Q is not None:
            QI = partition.Q.getConstantAs('MeV')
        else:
            QI = reaction.getQ('MeV')
        if p == projectile4LabEnergies:
            p4LE = PoPs[p].getMass('amu');   t4LE = PoPs[t].getMass('amu'); 
            lab2cmd = t4LE / (p4LE + t4LE)
            Qvalued = QI
        if p == projectile.id:
            Qvaluei = QI
            
            
#     print('lab2cmi:',lab2cmi,'and lab2cmd:',lab2cmd)
    if args.exclude is not None:
        print('Exclude any data line with these substrings:',' '.join(args.exclude))
    EminFound = 1e6; EmaxFound = -1e6
    if args.emin is None and args.EMAX is None and args.exclude is None:
        data_lines = f.readlines( )
        n_data = len(data_lines)
        lines_excluded = 'No'
        lines_excluded = 0
    else:
        data_lines = []
        lines_excluded= 0   
        n_data = 0   
        for line in f.readlines():
            n_data += 1
            Ed = float(line.split()[0])# in lab frame of data file
            Ecm  = Ed*lab2cmd - Qvalued + Qvaluei # in cm frame of gnds projectile.
            includE = emin < Ecm < emax
            includN = args.exclude is None or not any(sub in line for sub in args.exclude)
            if includE and includN:
                data_lines.append(line)  
                EminFound = min(EminFound,Ecm)
                EmaxFound = max(EmaxFound,Ecm)
            else:
                lines_excluded += 1      
        
    print(n_data-lines_excluded,'data lines after -x and lab energies defined by projectile',projectile4LabEnergies,'(',lines_excluded,'lines excluded)')
    if EminFound < EmaxFound: print('Kept data in the Ecm g-p range [',EminFound,',',EmaxFound,'] using Qd,Qi =',Qvalued,Qvaluei,'\n')
    if args.maxData is not None: 
        if args.maxData < 0:
            data_lines = data_lines[:abs(args.maxData)]
        else:
            data_lines = numpy.random.choice(data_lines,args.maxData)
            print('Data size reduced from',n_data,'to',len(data_lines))
    f.close( )
#     angular_lines = sorted(data_lines, key=lambda x: (float(x.split()[1])>=0.)  )
    angular_lines = [ x for x in data_lines if float(x.split()[1])>=0. ] 
    tot_lines     = [ x for x in data_lines if x.split()[4]=='TOT' ] 
    aint_lines    = [ x for x in data_lines if float(x.split()[1])<0. and x.split()[4]!='TOT'] 
#     print('Angulars, aints, totals=',len(angular_lines),len(aint_lines),len(tot_lines) )
    n_angular = len(angular_lines)
    if args.anglesData is not None: 
        if args.anglesData < 0:
            angular_lines = angular_lines[:abs(args.anglesData)]
        else:
            angular_lines = list(numpy.random.choice(angular_lines,args.anglesData))
            print('Angular data size reduced from',n_angular ,'to',len(angular_lines))
    f.close( )    
    data_lines = angular_lines + aint_lines + tot_lines
    
    data_lines = sorted(data_lines, key=lambda x: (float(x.split()[1])<0.,x.split()[4]=='TOT',float(x.split()[0]), float(x.split()[1]) ) )
              
    dataFilter = ''
    if args.emin       is not None: dataFilter += '-e%s' % args.emin
    if args.EMAX       is not None: dataFilter += '-E%s' % args.EMAX
    if args.maxData    is not None: dataFilter += '_m%s' % args.maxData
    if args.anglesData is not None: dataFilter += '_a%s' % args.anglesData
    
    if dataFilter != '':
        with open(args.dataFile.replace('.data',dataFilter+'.data')+'2','w') as fout: fout.writelines([projectile4LabEnergies+'\n'] + data_lines)
    
    n_data = len(data_lines)
    data_val = numpy.zeros([n_data,5], dtype=REAL)    # Elab,mu, datum,absError
    data_p   = numpy.zeros([n_data,2], dtype=INT)    # pin,pout
    
    groups = set()
    X4groups = set()
    group_list   = []
    cluster_list = []
    Ein_list = []
    Aex_list = []
    id = 0
    n_angles = 0
    n_angle_integrals = 0
    ni = 0
    for l in data_lines:
        parts = l.split()

        if len(parts)!=13: 
            print('Incorrect number of items in',l)
            sys.exit()
        Elab,CMangle,projectile,target,ejectile,residual,datum,absError,ex2cm,group,cluster,Ein,Aex = parts
        Elab,CMangle,datum,absError = float(Elab),float(CMangle),float(datum),float(absError)
        ex2cm,Aex = float(ex2cm),float(Aex)
#         print('p,t,e,r =',projectile,target,ejectile,residual)
        inLabel = projectile + " + " + target
        outLabel = ejectile + " + " + residual
        if outLabel == ' + ': outLabel = inLabel   # elastic
        pin = partitions.get(inLabel,None)
        pout= partitions.get(outLabel,None) if ejectile != 'TOT' else -1
        if pin is None:
            print("Entrance partition",inLabel,"not found in list",partitions.keys(),'in line',l)
            sys.exit()
        if pout is None:
            print("Exit partition",outLabel,"not found in list",partitions.keys(),'in line',l)
            sys.exit()
        
        thrad = CMangle*pi/180.
        mu = math.cos(thrad)
        if thrad < 0 : mu = 2   # indicated angle-integrated cross-section data
        if pout == -1: mu =-2   # indicated total cross-section data
        group_list.append(group)
        cluster_list.append(cluster)
        Ein_list.append(Ein)
        Aex_list.append(Aex)
        data_val[id,:] = [Elab,mu, datum,absError,ex2cm]
        data_p[id,:] = [pin,pout]
        groups.add(group)
        X4group = group.split('@')[0] + '@'
        X4groups.add(X4group)
        
        if CMangle > 0:  n_angles = id + 1  # number of angle-data points
        if CMangle < 0 and ejectile != 'TOT': n_angle_integrals = id+1  - n_angles  # number of Angle-ints after the angulars
        id += 1
    
#     if not args.norm1: print('Fitted norms:',Fitted_norm)
    f = open( args.normFile )
    norm_lines = f.readlines( )
    f.close( )    
    n_norms= len(norm_lines)
    norm_val = numpy.zeros(n_norms, dtype=REAL)  # norm,step,expect,syserror
    norm_info = numpy.zeros([n_norms,3], dtype=REAL)  # norm,step,expect,syserror
    norm_refs= []    
    ni = 0
    n_cnorms = 0
    n_fixed = 0
    n_free = 0
    tempfixes = 0
    for l in norm_lines:
        parts = l.split()
#         print(parts)
        norm,step, name,expect,syserror,reffile = parts
        norm,step,expect,syserror = float(norm),float(step),float(expect),float(syserror)

        fitted = Fitted_norm.get(name,None)
#         print('For name',name,'find',fitted)
        if fitted is not None and not args.norm1:
            print("Using previously fitted norm for %-23s: %12.5e instead of %12.5e" % (name,fitted,norm) )
            norm = fitted
        norm_val[ni] = norm
        if syserror > 0.:   # fitted norm
            chi_scale = 1.0/syserror 
            if args.normsfixed:
                fixed = 1
                chi_scale = 0.
                tempfixes += 1
            else:
                fixed = 0
                n_cnorms += 1
        elif syserror < 0.: # free norm
            fixed = 0
            chi_scale = 0
            n_free += 1
        else:               # fixed norm
            fixed = 1
            chi_scale = 0
            n_fixed += 1
        norm_info[ni,:] = (expect,chi_scale,fixed)
        norm_refs.append([name,reffile])
        ni += 1

    n_totals = n_data - n_angles - n_angle_integrals
    print('\nData points:',n_data,'of which',n_angles,'are for angles,',n_angle_integrals,'are for angle-integrals, and ',n_totals,'are for totals',
          '\nData groups:',len(groups),'\nX4 groups:',len(X4groups),
          '\nVariable norms:',n_norms,' of which ',n_cnorms,'constrained,',n_free,'free, and',n_fixed,' fixed (',tempfixes,'temporarily)\n')

    if dataFilter != '':
        with open(args.normFile.replace('.norms',dataFilter+'.norms')+'2','w') as fout: fout.writelines(norm_lines)
    
    effect_norm = numpy.zeros([n_data,n_norms], dtype=REAL)
    for ni in range(n_norms):
        reffile = norm_refs[ni][1]
        pattern = re.compile(reffile)
        for id in range(n_data):
            matching = pattern.match(group_list[id])
            effect_norm[id,ni] = 1.0 if matching else 0.0
#             if matching and args.debug: 
#                 print('Pattern',reffile,' ? ',group_list[id],':', matching)
    if args.debug:
        for ni in range(n_norms):
            print('norm_val[%i]' % ni,norm_val[ni],norm_info[ni,:])
#         for id in range(n_data):
#             print('VN for id',id,':',effect_norm[id,:])

    if args.Fixed is not None: 
        print('Fixed variables:',args.Fixed)
    else:
        args.Fixed = []
    print('Energy limits:   Data min,max:',args.emin,args.EMAX,'.  Poles min,max:',args.pmin,args.PMAX)

    finalStyleName = 'fitted'
    fitStyle = stylesModule.crossSectionReconstructed( finalStyleName,
            derivedFrom=gnd.styles.getEvaluatedStyle().label )

# parameter input for computer method
    base = args.inFile
    if args.single:           base += 's'
    if args.Multi>0:          base += 'm%s' % args.Multi
    if args.Grid != Gdefault: base += '+G%s' % args.Grid
# data input
    base += '+%s' % args.dataFile.replace('.data','')
    base += dataFilter
# searching
    if len(args.Fixed) > 0:         base += '_Fix:' + ('+'.join(args.Fixed)).replace('.*','@').replace('[',':').replace(']',':')
    if args.normsfixed            : base += '+n' 
    if args.pmin       is not None: base += '-p%s' % args.pmin
    if args.PMAX       is not None: base += '-P%s' % args.PMAX
    if args.dmin       is not None: base += '-d%s' % args.dmin
    if args.DMAX       is not None: base += '-D%s' % args.DMAX
    if args.Lambda     is not None: base += '-L%s' % args.Lambda
    if args.init       is not None: base += '@i%s'  % args.init[0]
    if args.init       is not None: print('Re-initialize at line',args.init[0],'of snap file',args.init[1])
    if args.Search     is not None: base += '+S%s'  % args.Search +  '_I%s' % args.Iterations
    if args.widthWeight is not None and args.widthWeight != 0.0: 
        base += ('_w%s' % args.widthWeight).replace('.0','')
    if args.Cross_Sections: base += '+C'
    if args.tag != '': base = base + '_'+args.tag

     
    dataDir = base 
#   if args.Cross_S0ctions or args.Matplot or args.TransitionMatrix >= 0 : os.system('mkdir '+dataDir)
    if args.Search or args.Cross_Sections : os.system('mkdir '+dataDir)
    print("Finish setup: ",tim.toString( ))
 
    chisq,ww,xsc,norm_val,n_pars,n_dof,XS_totals,ch_info,cov  = Gflow(
                        gnd,partitions,base,projectile4LabEnergies,data_val,data_p,n_angles,n_angle_integrals,
                        Ein_list,args.Fixed,args.emin,args.EMAX,args.pmin,args.PMAX,args.dmin,args.DMAX,args.Multi,args.ABES,args.Grid,
                        norm_val,norm_info,norm_refs,effect_norm, args.Lambda,args.LMatrix,args.groupAngles,
                        args.init,args.Search,args.Iterations,args.widthWeight,args.restarts,args.Background,args.BG,args.ReichMoore,  
                        args.Cross_Sections,args.verbose,args.debug,args.inFile,fitStyle,'_'+args.tag,args.Large,ComputerPrecisions,tim)

#     print("Finish rflow call: ",tim.toString( ))
#     print('\n ChiSq/pt = %10.4f from %i points' % (chisqppt,n_data))

    if args.Search:  
    
        print('Revised norms:',norm_val)
        saveNorms2gnds(gnd,docData,previousFit,computerCodeFit,n_norms,norm_val,norm_refs)

        info = '+S_' + args.tag
        newFitFile = base  + '-fit.xml'
        open( newFitFile, mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )
        print('Written new fit file:',newFitFile)
        
        if cov is not None:
            newCovFile = base  + '-fit_covs.xml'
            open( newCovFile, mode='w' ).writelines( line+'\n' for line in cov.toXMLList( ) )            
    else:
        info = '' 

    if args.Cross_Sections:
    
        XSp_tot_n,XSp_cap_n,XSp_mat_n = XS_totals
        pname,tname, za,zb, npairs,cm2lab,QI,ipair = ch_info

        EIndex = numpy.argsort(data_val[:,0])
        if args.TransitionMatrix >= 0:
            pnin,unused = printExcitationFunctions(XSp_tot_n,XSp_cap_n,XSp_mat_n, pname,tname, za,zb, npairs, base+'/'+base,n_data,data_val[:,0],EIndex,cm2lab,QI,ipair,True)
            pnin,totals = printExcitationFunctions(XSp_tot_n,XSp_cap_n,XSp_mat_n, pname,tname, za,zb, npairs, base+'/'+base,n_data,data_val[:,0],EIndex,cm2lab,QI,ipair,False)
            pnin = 'for %s' % pnin
        else:
            totals = None
            pnin = ''

        plotOut(n_data,n_norms,n_dof,args, base,info,dataDir, 
            chisq,ww,data_val,norm_val,norm_info,effect_norm,norm_refs, previousFit,computerCodeFit,
            groups,cluster_list,group_list,Ein_list,Aex_list,xsc,X4groups, data_p,pins, args.TransitionMatrix,
            args.XCLUDE,projectile4LabEnergies,data_lines,args.dataFile,
            EIndex,totals,pname,tname,args.datasize,ipair,cm2lab, emin,emax,pnin,gnd,cmd )
    
    print('\n*** chisq/pt = %12.5f, chisq/dof = %12.5f with ww = %12.5f so data Chisq/DOF = %12.5f from dof = %i\n' % (chisq/n_data,chisq/n_dof,ww/n_dof,(chisq-ww)/n_dof, n_dof) )   
    print("Final rflow: ",tim.toString( ))
    print("Target stdout:",dataDir + '.out')
