
import times
tim = times.times()

import os,math,numpy,cmath,pwd,sys,time,json,re

from CoulCF import cf1,cf2,csigma,Pole_Shifts
from opticals import get_optical_S

from pqu import PQU as PQUModule
from numericalFunctions import angularMomentumCoupling
from xData.series1d  import Legendre
from xData import XYs
from PoPs.groups.misc import *

from xData.Documentation import documentation as documentationModule
from xData.Documentation import computerCode  as computerCodeModule
# from scipy import interpolate

REAL = numpy.double
CMPLX = numpy.complex128
INT = numpy.int32

# CONSTANTS: 
hbc =   197.3269788e0             # hcross * c (MeV.fm)
finec = 137.035999139e0           # 1/alpha (fine-structure constant)
amu   = 931.4940954e0             # 1 amu/c^2 in MeV

coulcn = hbc/finec                # e^2
fmscal = 2e0 * amu / hbc**2
etacns = coulcn * math.sqrt(fmscal) * 0.5e0
pi = 4.*math.atan(1.0)
rsqr4pi = 1.0/(4*pi)**0.5

def nuclIDs (nucl):
    datas = chemicalElementALevelIDsAndAnti(nucl)
    if datas[1] is not None:
        return datas[1]+str(datas[2]),datas[3]
    else:
        return datas[0],0

lightnuclei = {'n':'n', 'H1':'p', 'H2':'d', 'H3':'t', 'He3':'h', 'He4':'a', 'photon':'g'}

def quickName(p,t):     #   (He4,Be11_e3) -> a3
    ln = lightnuclei.get(p,p)
    tnucl,tlevel = nuclIDs(t)
    qn = ln + str(tlevel) if tlevel>0 else ln
    return(qn,tlevel)

def generateEnergyGrid(energies,widths, lowBound, highBound, stride=1):
    """ Create an initial energy grid by merging a rough mesh for the entire region (~10 points / decade)
    with a denser grid around each resonance. For the denser grid, multiply the total resonance width by
    the 'resonancePos' array defined below. """
    thresholds = []
    # ignore negative resonances
    for lidx in range(len(energies)):
        if energies[lidx] > 0: break
    energies = energies[lidx:]
    widths = widths[lidx:]
    # generate grid for a single peak, should be good to 1% using linear interpolation using default stride
    resonancePos = numpy.array([
        5.000e-04, 1.000e-03, 2.000e-03, 3.000e-03, 4.000e-03, 5.000e-03, 6.000e-03, 7.000e-03, 8.000e-03, 9.000e-03, 1.000e-02, 2.000e-02,
        3.000e-02, 4.000e-02, 5.000e-02, 6.000e-02, 7.000e-02, 8.000e-02, 9.000e-02, 1.000e-01, 1.100e-01, 1.200e-01, 1.300e-01, 1.400e-01,
        1.500e-01, 1.600e-01, 1.700e-01, 1.800e-01, 1.900e-01, 2.000e-01, 2.100e-01, 2.200e-01, 2.300e-01, 2.400e-01, 2.500e-01, 2.600e-01,
        2.800e-01, 3.000e-01, 3.200e-01, 3.400e-01, 3.600e-01, 3.800e-01, 4.000e-01, 4.200e-01, 4.400e-01, 4.600e-01, 4.800e-01, 5.000e-01,
        5.500e-01, 6.000e-01, 6.500e-01, 7.000e-01, 7.500e-01, 8.000e-01, 8.500e-01, 9.000e-01, 9.500e-01, 1.000e+00, 1.050e+00, 1.100e+00,
        1.150e+00, 1.200e+00, 1.250e+00, 1.300e+00, 1.350e+00, 1.400e+00, 1.450e+00, 1.500e+00, 1.550e+00, 1.600e+00, 1.650e+00, 1.700e+00,
        1.750e+00, 1.800e+00, 1.850e+00, 1.900e+00, 1.950e+00, 2.000e+00, 2.050e+00, 2.100e+00, 2.150e+00, 2.200e+00, 2.250e+00, 2.300e+00,
        2.350e+00, 2.400e+00, 2.450e+00, 2.500e+00, 2.600e+00, 2.700e+00, 2.800e+00, 2.900e+00, 3.000e+00, 3.100e+00, 3.200e+00, 3.300e+00,
        3.400e+00, 3.600e+00, 3.800e+00, 4.000e+00, 4.200e+00, 4.400e+00, 4.600e+00, 4.800e+00, 5.000e+00, 5.200e+00, 5.400e+00, 5.600e+00,
        5.800e+00, 6.000e+00, 6.200e+00, 6.400e+00, 6.500e+00, 6.800e+00, 7.000e+00, 7.500e+00, 8.000e+00, 8.500e+00, 9.000e+00, 9.500e+00,
        1.000e+01, 1.050e+01, 1.100e+01, 1.150e+01, 1.200e+01, 1.250e+01, 1.300e+01, 1.350e+01, 1.400e+01, 1.450e+01, 1.500e+01, 1.550e+01,
        1.600e+01, 1.700e+01, 1.800e+01, 1.900e+01, 2.000e+01, 2.100e+01, 2.200e+01, 2.300e+01, 2.400e+01, 2.500e+01, 2.600e+01, 2.700e+01,
        2.800e+01, 2.900e+01, 3.000e+01, 3.100e+01, 3.200e+01, 3.300e+01, 3.400e+01, 3.600e+01, 3.800e+01, 4.000e+01, 4.200e+01, 4.400e+01,
        4.600e+01, 4.800e+01, 5.000e+01, 5.300e+01, 5.600e+01, 5.900e+01, 6.200e+01, 6.600e+01, 7.000e+01, 7.400e+01, 7.800e+01, 8.200e+01,
        8.600e+01, 9.000e+01, 9.400e+01, 9.800e+01, 1.020e+02, 1.060e+02, 1.098e+02, 1.140e+02, 1.180e+02, 1.232e+02, 1.260e+02, 1.300e+02,
        1.382e+02, 1.550e+02, 1.600e+02, 1.739e+02, 1.800e+02, 1.951e+02, 2.000e+02, 2.100e+02, 2.189e+02, 2.300e+02, 2.456e+02, 2.500e+02,
        2.600e+02, 2.756e+02, 3.092e+02, 3.200e+02, 3.469e+02, 3.600e+02, 3.892e+02, 4.000e+02, 4.200e+02, 4.367e+02, 4.600e+02, 4.800e+02,
        5.000e+02, 6.000e+02, 7.000e+02, 8.000e+02, 9.000e+02, 1.000e+03, 1.020e+03, 1.098e+03, 1.140e+03, 1.232e+03, 1.260e+03, 1.300e+03,
        1.382e+03, 1.550e+03, 1.600e+03, 1.739e+03, 1.800e+03, 1.951e+03, 2.000e+03, 2.100e+03, 2.189e+03, 2.300e+03, 2.456e+03, 2.500e+03,
        2.600e+03, 2.756e+03, 3.092e+03, 3.200e+03, 3.469e+03, 3.600e+03, 3.892e+03, 4.000e+03, 4.200e+03, 4.367e+03, 4.600e+03, 4.800e+03,
        5.000e+03, 6.000e+03, 7.000e+03, 8.000e+03, 9.000e+03, 1.000e+04
         ][::stride])

    grid = []
    # get the midpoints (on log10 scale) between each resonance:
    # emid = [lowBound] + list(10**( ( numpy.log10(energies[1:])+numpy.log10(energies[:-1]) ) / 2.0)) + [highBound]
    # or get midpoints on linear scale:
    emid = [lowBound] + [(e1+e2)/2.0 for e1, e2 in zip(energies[1:], energies[:-1])] + [highBound]
    for e, w, lowedge, highedge in zip(energies, widths, emid[:-1], emid[1:]):
        points = e-w*resonancePos
        grid += [lowedge] + list(points[points>lowedge])
#         print('Around e,w=',e,w,': below:',list(points[points>lowedge]))
        points = e+w*resonancePos[1:]
        grid += list(points[points < highedge])
#         print('Around e,w=',e,w,': aboveG:',list(points[points < highedge]))
    # also add rough grid, to cover any big gaps between resonances, should give at least 10 points per decade:
    npoints = int(numpy.ceil(numpy.log10(highBound)-numpy.log10(lowBound)) * 10)
    grid += list(numpy.logspace(numpy.log10(lowBound), numpy.log10(highBound), npoints))[1:-1]
    grid += [lowBound, highBound, 0.0253]   # region boundaries + thermal
    # if threshold reactions present, add dense grid at and above threshold
    for threshold in thresholds:
        grid += [threshold]
        grid += list(threshold + resonancePos * 1e-2)
    grid = sorted(set(grid))
    # toss any points outside of energy bounds:
    grid = grid[grid.index(lowBound) : grid.index(highBound)+1]
    return numpy.asarray(grid, dtype=REAL)
                       

def Gomp(gnds,base,emin,emax,jmin,jmax,Dspacing,optical_potentials,Model,YRAST, hcm,offset,Convolute,stride,
       verbose,debug,inFile,ComputerPrecisions,tim):
        
    REAL, CMPLX, INT, realSize = ComputerPrecisions

    PoPs = gnds.PoPs
    projectile = PoPs[gnds.projectile]
    target     = PoPs[gnds.target]
    elasticChannel = '%s + %s' % (gnds.projectile,gnds.target)
    if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
    if hasattr(target, 'nucleus'):     target = target.nucleus
    pZ = projectile.charge[0].value; tZ =  target.charge[0].value
#     rStyle = fitStyle.label
    
    rrr = gnds.resonances.resolved
    Rm_Radius = gnds.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    bndx = RMatrix.boundaryCondition
    IFG = RMatrix.reducedWidthAmplitudes
    Overrides = False
    brune = bndx=='Brune'
    if brune: LMatrix = True
#     if brune and not LMatrix:
#         print('Brune basis requires Level-matrix method')
#         LMatrix = True
 
    n_jsets = len(RMatrix.spinGroups)
    n_poles = 0
    n_chans = 0
    
    np = len(RMatrix.resonanceReactions)
    partitions = {}
    ReichMoore = False
    for pair in range(np):
#         print('Partition',pair,'elim,label',RMatrix.resonanceReactions[pair].eliminated,RMatrix.resonanceReactions[pair].label)
        kp = RMatrix.resonanceReactions[pair].label
        partitions[kp] = pair
        if RMatrix.resonanceReactions[pair].eliminated: 
            ReichMoore = True
            damping_pair = pair
            damping_label = kp
            print('\nReich-Moore channel is',damping_pair,':',damping_label)
        
    prmax = numpy.zeros(np, dtype=REAL)
    QI = numpy.zeros(np, dtype=REAL)
    rmass = numpy.zeros(np, dtype=REAL)
    za = numpy.zeros(np, dtype=REAL)
    zb = numpy.zeros(np, dtype=REAL)
    jp = numpy.zeros(np, dtype=REAL)
    pt = numpy.zeros(np, dtype=REAL)
    ep = numpy.zeros(np, dtype=REAL)
    jt = numpy.zeros(np, dtype=REAL)
    tt = numpy.zeros(np, dtype=REAL)
    et = numpy.zeros(np, dtype=REAL)
    AT = numpy.zeros(np, dtype=REAL)
    cm2lab  = numpy.zeros(np, dtype=REAL)
    pname = ['' for i in range(np)]
    tname = ['' for i in range(np)]

    channels = {}
    pair = 0
    inpair = None
    chargedElastic = False
    OpticalPot = []
    print('\nChannels:')
    for partition in RMatrix.resonanceReactions:
        kp = partition.label
        if partition.eliminated: # no two-body kinematics
            partitions[kp] = None
            continue
        channels[pair] = kp
        reaction = partition.reactionLink.link
        p,t = partition.ejectile,partition.residual
        pname[pair] = p
        tname[pair] = t
        projectile = PoPs[p];
        target     = PoPs[t];
        A_B = '%s + %s' % (p,t)
        if Dspacing is not None:
            if A_B not in optical_potentials.keys():
                print('\nChannel',A_B,'optical potential not specified. Stop')
                sys.exit()
            else:
                OpticalPot.append(optical_potentials[A_B])

        pMass = projectile.getMass('amu');   
        AT[pair] = target.A
        tMass =     target.getMass('amu');
        rmass[pair] = pMass * tMass / (pMass + tMass)
        if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
        if hasattr(target, 'nucleus'):     target = target.nucleus

        za[pair]    = projectile.charge[0].value;  
        zb[pair]  = target.charge[0].value
        if za[pair]*zb[pair] != 0: chargedElastic = True
        if partition.Q is not None: 
            QI[pair] = partition.Q.getConstantAs('MeV')
        else:
            QI[pair] = reaction.getQ('MeV')
        if partition.scatteringRadius is not None:
            prmax[pair] =  partition.scatteringRadius.getValueAs('fm')
        else:
            prmax[pair] = Rm_global

        if partition.label == elasticChannel:
            ipair = pair  # incoming
        cm2lab[pair] = (pMass + tMass) / tMass
            
        jp[pair],pt[pair],ep[pair] = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try:
            jt[pair],tt[pair],et[pair] = target.spin[0].float('hbar'), target.parity[0].value, target.energy[0].pqu('MeV').value
        except:
            jt[pair],tt[pair],et[pair] = 0.,1,0.
        print(pair,":",kp,' Q =',QI[pair],'R =',prmax[pair],' spins',jp[pair],jt[pair])
        pair += 1
    lab2cm = 1.0/cm2lab[ipair]
    
    if verbose: print("\nElastic channel is",elasticChannel,'with IFG=',IFG)
#     if debug: print("Charged-particle elastic:",chargedElastic,",  identical:",identicalParticles,' rStyle:',rStyle)
    npairs  = pair
    if not IFG:
        print("Not yet coded for IFG =",IFG)
        sys.exit()
    
#  FIRST: for array sizes:
#

    Lmax = 0
    for Jpi in RMatrix.spinGroups:
        R = Jpi.resonanceParameters.table
        n_poles = max(n_poles,R.nRows)
        n = R.nColumns-1
        if ReichMoore: n -= 1
        n_chans = max(n_chans,n)
        for ch in Jpi.channels:
            Lmax = max(Lmax,ch.L)
    print('Initially %i Jpi sets with %i poles max, and %i channels max. Lmax=%i' % (n_jsets,n_poles,n_chans,Lmax))
    
    N_opts = max(int( (emax - emin)/Dspacing + 1.5), 0) if Dspacing is not None else 0
    D = (emax-emin)/(N_opts-1) if N_opts > 1 else 0.0
    print("Increase max poles from",n_poles,"towards",n_poles + N_opts)
    n_poles += N_opts

    nch = numpy.zeros(n_jsets, dtype=INT)
    npli = numpy.zeros(n_jsets, dtype=INT)
    npl = numpy.zeros(n_jsets, dtype=INT)
    E_poles = numpy.zeros([n_jsets,n_poles], dtype=REAL) + 1e6
    D_poles = numpy.zeros([n_jsets,n_poles], dtype=REAL)
    has_widths = numpy.zeros([n_jsets,n_poles], dtype=INT)
    g_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=REAL)
    
    GNDS_order = numpy.zeros([n_jsets,n_poles,n_chans+2], dtype=INT) # [,,0] is energy, 1 is damping, as in GNDS
    J_set = numpy.zeros(n_jsets, dtype=REAL)
    pi_set = numpy.zeros(n_jsets, dtype=INT)
    L_val  =  numpy.zeros([n_jsets,n_chans], dtype=INT)
    S_val  =  numpy.zeros([n_jsets,n_chans], dtype=REAL)
    B_val  =  numpy.zeros([npairs,n_jsets,n_chans], dtype=REAL)
    p_mask =  numpy.zeros([npairs,n_jsets,n_chans], dtype=REAL)
    seg_val=  numpy.zeros([n_jsets,n_chans], dtype=INT) - 1 
    seg_col=  numpy.zeros([n_jsets], dtype=INT) 

    Spins = [set() for pair in range(npairs)]

    
## DATA
# Indexing to channels, etc
    jset = 0
    tot_channels = 0; tot_poles = 0
    all_partition_channels = numpy.zeros(npairs, dtype=INT)
    c0 = numpy.zeros([n_jsets,npairs], dtype=INT)
    cn = numpy.zeros([n_jsets,npairs], dtype=INT)
    All_spins = set()
    for Jpi in RMatrix.spinGroups:
        J_set[jset] = Jpi.spin
        pi_set[jset] = Jpi.parity
        parity = '+' if pi_set[jset] > 0 else '-'
        R = Jpi.resonanceParameters.table
        cols = R.nColumns - 1  # ignore energy col
        if ReichMoore: cols -= 1 # ignore damping width just now
        rows = R.nRows
        nch[jset] = cols
        npli[jset] = rows
        npl[jset] = rows + N_opts
        if True: print('J,pi =%5.1f %s, channels %3i, poles %3i -> %i' % (J_set[jset],parity,cols,npli[jset],npl[jset]) )
        tot_channels += cols
        tot_poles    += rows
        seg_col[jset] = cols    

        c = 0
        partition_channels = numpy.zeros(npairs, dtype=INT)
        for pair in range(npairs):
            c0[jset,pair] = c
            for ch in Jpi.channels:
                rr = ch.resonanceReaction
                pairc = partitions.get(rr,None)
                if pairc is None or pairc!=pair: continue
                m = ch.columnIndex - 1
                L_val[jset,c] = ch.L
                S = float(ch.channelSpin)
                S_val[jset,c] = S
                has_widths[jset,:rows] = 1
            
                seg_val[jset,c] = pair
                p_mask[pair,jset,c] = 1.0
                partition_channels[pair] += 1
            
                Spins[pair].add(S)
                All_spins.add(S)    
    
                c += 1
            cn[jset,pair] = c
        all_partition_channels = numpy.maximum(all_partition_channels,partition_channels)
        jset += 1   

    print(' Total channels',tot_channels,' and total poles',tot_poles,'\n')
    maxpc = numpy.amax(all_partition_channels)
    print('Max channels in each partition:',all_partition_channels,' max=',maxpc)
    for jset in range(n_jsets):
        if debug: print('Channel ranges for each parition:',[[c0[jset,pair],cn[jset,pair]] for pair in range(npairs)])
        
    if debug:
        print('All spins:',All_spins)
        print('All channel spins',Spins)
                
            
#  SECOND: fill in arrays:
    jset = 0
    G_order = 0 
    for Jpi in RMatrix.spinGroups:
        R = Jpi.resonanceParameters.table
#         print('R:\n',R.data)
        J = J_set[jset]
        cols = R.nColumns - 1  # ignore energy col
        rows = R.nRows
        E_poles[jset,:rows] = numpy.asarray( R.getColumn('energy','MeV') , dtype=REAL)   # lab MeV
        ncols = cols + 1
        if ReichMoore: 
#             D_poles[jset,:rows] = numpy.asarray( R.getColumn(damping_label,'MeV') , dtype=REAL)   # lab MeV
            for n in range(rows):
                D_poles[jset,n] = R.data[n][damping_pair+1]
            if IFG==1:     D_poles[jset,:] = 2*D_poles[jset,:]**2
        if jmin <= J <= jmax: 
            for ie in range(N_opts):
                e = emin + D * (ie + offset * (Jpi.spin + int(Jpi.parity)/3.0) )  + YRAST * J*(J+1.)
                if e > emax: continue
                n = npli[jset]+ie
                E_poles[jset,n] = e
                has_widths[jset,n] = 1
                rows += 1
        npl[jset] = rows
        
        for n in range(rows):
            for c in range(ncols):
                GNDS_order[jset,n,c] = G_order  # order of variables in GNDS and ENDF, needed for covariance matrix
                G_order += 1 

        widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']
        
#         if verbose:  print("\n".join(R.toXMLList()))       
        n = None
        c = 0
        for pair in range(npairs):
            for ch in Jpi.channels:
                ic = c - c0[jset,pair]
                rr = ch.resonanceReaction
                pairc = partitions.get(rr,None)
                if pairc is None or pairc!=pair: continue
                m = ch.columnIndex - 1
                g_poles[jset,:npli[jset],c] = numpy.asarray(widths[m][:],  dtype=REAL) 

                c += 1
        if debug:
            print('J set %i: E_poles \n' % jset,E_poles[jset,:])
            print('g_poles \n',g_poles[jset,:,:])
        jset += 1   

    L_poles = numpy.zeros([n_jsets,n_poles,n_chans,2], dtype=REAL)
    dLdE_poles = numpy.zeros([n_jsets,n_poles,n_chans,2], dtype=REAL)
    Pole_Shifts(L_poles,dLdE_poles, E_poles,has_widths, seg_val,1./cm2lab[ipair],QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
    print()
        
        
    if Dspacing is not None:
    # CALCULATE OPTICAL-MODEL SCATTERING TO GET PARTIAL WIDTHS

        omfile = open(base + '-omp.txt','w')
        sc_info = []
        ncm = int( prmax[ipair] / hcm + 0.5)
        hcm = prmax[ipair]/ncm
        isc = 0
        for jset,Jpi in enumerate(RMatrix.spinGroups):
            parity = '+' if pi_set[jset] > 0 else '-'
            R = Jpi.resonanceParameters.table
            cols = R.nColumns - 1  # ignore energy col
            isc_i = isc
            for c,ch in enumerate(Jpi.channels):
                L =  L_val[jset,c]
                S =  S_val[jset,c]
                for ie in range(N_opts):
                    n = npli[jset]+ie
                    e = E_poles[jset,n]

                    pair = seg_val[jset,c] 
                    if pair < 0 or e > emax: continue
                    E = e*lab2cm + QI[pair]
                    if E <= 1e-3: continue
                
                    sqE = math.sqrt(E)
                    a = prmax[pair]
                    h = a / ncm
                    rho = a * math.sqrt(fmscal*rmass[pair]) * sqE
                    eta = etacns * za[pair]*zb[pair] * math.sqrt(rmass[pair]) / sqE 
                    EPS=1e-10; LIMIT = 2000000; ACC8 = 1e-12
                    F = cf1(rho,eta,L,EPS,LIMIT) * rho
                    Shift = L_poles[jset,n,c,0]
                    P     = L_poles[jset,n,c,1]
                    phi = - math.atan2(P, F - Shift)
                    sc_info.append([jset,c,n,h,L,S,pair,E,a,rmass[pair],pname[pair],za[pair],zb[pair],AT[pair],L_poles[jset,n,c,:],phi, OpticalPot[pair]])

                    print( jset,c,ie,  'j,c,e Scatter',pname[pair],'on',tname[pair],'at E=',E,'LS=',L,S,'with',OpticalPot[pair], file=omfile)
                    isc += 1
            print('J,pi =%5.1f %s, channels %3i, widths %5i -> %5i (incl)' % (J_set[jset],parity,cols,isc_i,isc-1))
    
        Smat = get_optical_S(sc_info,ncm, omfile)
        SmatMSQ = (Smat * numpy.conjugate(Smat)).real
        TC = 1.0 - SmatMSQ

        Dcm = Dspacing*lab2cm
        print('Model',Model)
        if   Model[0]=='A':
            AvFormalWidths = Dspacing * (-numpy.log(SmatMSQ)) / (2.*pi)
        elif Model[0]=='B':
            AvFormalWidths = Dspacing * TC / (2.*pi)
        else:
            print('Model',Model,'unrecognized')
            sys.exit()
        mparts = Model.split(',')
        scale = 1.0
        Eslope = 0.0
        if len(mparts)>1:
            scale = float(mparts[1])
            if len(mparts)>2: Eslope = float(mparts[2])
            
        for isc,sc in  enumerate(sc_info):
            jset,c,n = sc[:3]
            E = sc[7]
            AvFormalWidths[isc] *= scale  + E * Eslope 
            if E < 1e-3: continue    # sub-threshold 
            g_poles[jset,n,c] = AvFormalWidths[isc]
            if IFG==1:  # get rwa
                P =  L_poles[jset,n,c,1]
                g_poles[jset,n,c] = abs(AvFormalWidths[isc]/(2*P)) ** 0.5
                print(isc,jset,c,n,E,'fw=',AvFormalWidths[isc],'P=',P,'rwa=', g_poles[jset,n,c], file=omfile)
                
    # Copy parameters back into GNDS 
        jset = 0
        for Jpi in RMatrix.spinGroups:   # jset values need to be in the same order as before
            parity = '+' if pi_set[jset] > 0 else '-'
            if True: print('\nJ,pi =',J_set[jset],parity, file=omfile)
            R = Jpi.resonanceParameters.table
            rows = R.nRows
            cols = R.nColumns - 1  # without energy col
            c_start = 1
            if ReichMoore: 
                cols -= 1
                c_start = 2
            for pole in range(rows):
    #               print('Update',pole,'pole',R.data[pole][0],'to',E_poles[jset,pole])
                R.data[pole][0] = E_poles[jset,pole]
                if ReichMoore:
                    if IFG:
                        R.data[pole][1] = math.sqrt(D_poles[jset,pole]/2.)
                    else:
                        R.data[pole][1] = D_poles[jset,pole]
                for c in range(cols):
                    R.data[pole][c+c_start] = g_poles[jset,pole,c]
    #                 if verbose: print('\nJ,pi =',J_set[jset],parity,"revised R-matrix table:", "\n".join(R.toXMLList()))

            for ie in range(N_opts):
                n = npli[jset]+ie
                if E_poles[jset,n] > emax: break
                row = [ E_poles[jset,n] ]
                if ReichMoore: row.append(0.0)   # ReichMoore damping on new optical poles
                totwid = 0.0
                for c in range(cols): 
                    print('   Add optical width at j,n,c=',jset,n,c,':',g_poles[jset,n,c], file=omfile)
                    row.append(g_poles[jset,n,c])

                    if IFG==1:
                        totwid  += 2. * row[c+c_start]**2 *  L_poles[jset,n,c,1]
                    else:
                        totwid  +=  row[c+c_start] 
                        
                print('For',J_set[jset],parity,'pole',n,'at',E_poles[jset,n],' formal width=',totwid,'from', row[c+c_start] ,file=omfile)
                

                R.data.append(row)
            jset += 1
    
#     print('\nR-matrix parameters:\n')
#     for Jpi in RMatrix.spinGroups:   # jset values need to be in the same order as before
# #         parity = '+' if pi_set[jset] > 0 else '-'
#         R = Jpi.resonanceParameters.table
#         print('J,pi =',Jpi.spin,Jpi.parity,' matrix is',R.nRows,'*',R.nColumns, '(len=%i)' % len(R.data))

    print("Formal and 'Observed' properties by first-order (Thomas) corrections:")
#     L_poles    = numpy.zeros([n_jsets,n_poles,n_chans,2], dtype=REAL)
#     dLdE_poles = numpy.zeros([n_jsets,n_poles,n_chans,2], dtype=REAL)
#     Pole_Shifts(L_poles,dLdE_poles, E_poles,has_widths, seg_val,1./cm2lab[ipair],QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
    if brune:
        O_poles  = E_poles 
    else: # recalculate P and S' at the 'observed' energies
        O_poles  = E_poles - numpy.sum( g_poles**2 * L_poles[:,:,:,0], 2 ) * cm2lab[ipair]  # both terms lab
    Pole_Shifts(L_poles,dLdE_poles, O_poles,has_widths, seg_val,1./cm2lab[ipair],QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 

    F_widths = 2.* numpy.sum( g_poles**2 * L_poles[:,:,:,1], 2 )
    O_widths = F_widths / (1. + numpy.sum( g_poles**2 * dLdE_poles[:,:,:,0], 2 )) # cm
    TW = base + '.denoms'
    tw = open(TW,'w')
    
    energies = []
    Owidths = []
    for jset in range(n_jsets):
        print("  For J/pi = %.1f%s: %i"  % (J_set[jset],'+' if pi_set[jset] > 0 else '-',npl[jset]) )
        for p in range(npl[jset]):
            print('   Pole %3i at Elab = %10.6f (cm %10.6f, obs %10.6f) MeV widths: formal %10.5f, obs %10.5f, damping %10.5f' \
                  % (p,E_poles[jset,p],E_poles[jset,p]/cm2lab[ipair],O_poles[jset,p]/cm2lab[ipair],F_widths[jset,p],O_widths[jset,p],D_poles[jset,p]) )
            if abs(F_widths[jset,p]) < 1e-3: print(68*' ','widths: formal %10.3e, obs %10.3e' % (F_widths[jset,p],O_widths[jset,p]) )
            energies.append( O_poles [jset,p] )
            Owidths.append ( O_widths[jset,p] )
            if E_poles[jset,p]>0: 
               dd = 1.0 if Dspacing is None else Dcm
               print(E_poles[jset,p]/cm2lab[ipair],O_widths[jset,p]*2*pi/dd, file=tw)
        print('&',file=tw)

    print()
    noRecon = Convolute is None
    if noRecon and scale != 1.0 : return()
    
# calculate excitation cross-sections in MLBW formalism
    
    Global = False
    G = 'G' if Global else ''        

    E_scat = generateEnergyGrid(energies,Owidths, emin,emax, stride=stride)
    n_energies = len(E_scat)
    print('\nReconstruction energy grid over emin,emax =',emin,emax,'with',n_energies)

    if Convolute is not None and  Convolute > 0.0: 
        def spread(de,s):
            c = 1/pi**0.5 / s
            return (c* math.exp(-(de/s)**2))
    
        fun = []
        for i in range(100):
            de = (i-50)*Convolute*0.1
            f = spread(de,Convolute)
            fun.append([de,f])
        conv = XYs.XYs1d(fun)
        print("Convolute with Gaussian in %s * [-5,5] with steps of 0.1*%s" % (Convolute,Convolute))
    
    rksq_val  = numpy.zeros([n_energies,npairs], dtype=REAL)
    for pair in range(npairs):
        for ie in range(n_energies):
            E = E_scat[ie]*lab2cm + QI[pair]
            if abs(E) < 1e-10:
                E = (E + E_scat[ie+1]*lab2cm + QI[pair]) * 0.5
            rksq_val[ie,pair] = 1. / (fmscal * rmass[pair] * E)
            
    gfac = numpy.zeros([n_energies,n_jsets,n_chans])
    for jset in range(n_jsets):
        for c_in in range(n_chans):   # incoming partial wave
            pair = seg_val[jset,c_in]      # incoming partition
            if pair>=0:
                denom = (2.*jp[pair]+1.) * (2.*jt[pair]+1)
                for ie in range(n_energies):
                    gfac[ie,jset,c_in] = pi * (2*J_set[jset]+1) * rksq_val[ie,pair] / denom 
    
    Ex = numpy.zeros(n_energies)
    Ry = numpy.zeros(n_energies)
    Cy = numpy.zeros(n_energies)
    Dy = numpy.zeros(n_energies)
    
    F_width = 2.*  g_poles**2 * L_poles[:,:,:,1]   #   jset,p,c
    O_width = F_width / (1. + numpy.sum( g_poles**2 * dLdE_poles[:,:,:,0], 2, keepdims=True )) # cm

#     Pole_Shiftse(E_scat,L_polese,dLdE_poles, O_poles,has_widths, seg_val,1./cm2lab[ipair],QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
#     
#     F_widthe = 2.*  g_poles**2 * L_polese[:,:,:,1]   #   jset,p,c
#     O_widthe = F_widthe / (1. + numpy.sum( g_poles**2 * dLdE_E[:,:,:,0], 2, keepdims=True )) # cm    
    
#     for pin in range(npairs): 
#         for jset in range(n_jsets):
#             for p in range(npl[jset]):        
#                 print('J,pole:',jset,p,'F widths:',F_width[jset,p,:nch[jset]])
                                                    
    for pin in range(npairs):
        pn,il = quickName(pname[pin],tname[pin]) 
        if il>0: continue
        
        if Dspacing is not None:
            rname = base + '-MLBW-%sreac_%s' % (G,pn)
            hname = base + '-MLBW-%sCH_%s' % (G,pn)
            rout = open(rname,'w')
            hout = open(hname,'w')
            print('Reaction cross-sections for',pn,' to file   ',rname)
            print('Half reaction cross-sections for',pn,' to file   ',hname)

            
            for ie in range(N_opts):
                p = npli[jset]+ie
                e = emin + D * (ie + offset * (Jpi.spin + int(Jpi.parity)/3.0) )  # lab energy in ipair
                Ecm = e*lab2cm

                E = Ecm + QI[pin] - QI[ipair]   # cm energy in pin
                rk_isq = 1. / (fmscal * rmass[pin] * E)
                Elab = E * cm2lab[pin]   # Elab for incoming channel (pin, not ipair)
                denom = (2.*jp[pin]+1.) * (2.*jt[pin]+1)
            
                XSreac = 0.0
                for jset in range(n_jsets):
                    Gfac = pi * (2*J_set[jset]+1) * rk_isq / denom
                    for cin in range(nch[jset]):
                        if seg_val[jset,cin]!=pin: continue      
                        XSreac += Gfac * 2*pi * O_width[jset,p,cin] / Dspacing * 10.
                print(Elab,XSreac, file=rout)
                print(Elab,XSreac/2., file=hout)
            rout.close()
            hout.close()

        if noRecon: continue


        for pout in range(npairs):
            
            po,ol = quickName(pname[pout],tname[pout])
            fname = base + '-SLBW-%sch_%s-to-%s' % (G,pn,po)
            print('Partition',pn,'to',po,': angle-integrated cross-sections to file   ',fname)
            fout = open(fname,'w')
            fcname = base + '-NLBW-%scch_%s-to-%s' % (G,pn,po)
            print('Partition',pn,'to',po,': coherent angle-integrated cross-sections to file   ',fcname)
            fcout = open(fcname,'w')
            fdname = base + '-MLBW-%sdch_%s-to-%s' % (G,pn,po)
            print('Partition',pn,'to',po,': coherent (fixed-width) angle-integrated cross-sections to file   ',fdname)
            fdout = open(fdname,'w')
            sys.stdout.flush()
            
            for ie in range(n_energies):
                Ecm = E_scat[ie]*lab2cm  # pole energy in cm ipair.
                E = Ecm + QI[pin] - QI[ipair]

                XSp_mat = 0.
                XSp_coh = 0.
                XSa_coh = 0.
                
                for jset in range(n_jsets):
                    amp_coh = 0.0 + 0j
                    ama_coh = 0.0 + 0j
                    for p in range(npl[jset]):
                        if O_widths[jset,p] < 1e-5: continue
                        
                        for cin in range(nch[jset]):
                            if seg_val[jset,cin]!=pin: continue
                            if O_width[jset,p,cin] < 1e-5: continue
                            Lin = L_val[jset,cin]
                            for cout in range(nch[jset]):
                                if seg_val[jset,cout]!=pout: continue
                                if O_width[jset,p,cout] < 1e-5: continue
                                Lout = L_val[jset,cout]
                                
                                ampl  = cmath.sqrt( O_width[jset,p,cin] * O_width[jset,p,cout]  *  gfac[ie,jset,cin] )  \
                                     / complex( E_scat[ie] - E_poles[jset,p] , O_widths[jset,p]*0.5 + 1e-10)
                                amp_coh += ampl
                                
                                ampl2 = ampl * math.sqrt( max(0.,E_scat[ie]/E_poles[jset,p]) )  # ** (Lin + Lout + 1)
                                ama_coh += ampl2

                                XSp_mat += O_width[jset,p,cin] * O_width[jset,p,cout]  *  gfac[ie,jset,cin]  \
                                     / ( (E_scat[ie] - E_poles[jset,p])**2 + O_widths[jset,p]**2/4.0 )
                                     
                    XSp_coh +=  ( amp_coh * amp_coh.conjugate() ).real
                    XSa_coh +=  ( ama_coh * ama_coh.conjugate() ).real

                x = XSp_mat * 10.   # SLBW  decoherent
                xc = XSp_coh * 10.  # MLBW  fixed G
                xa = XSa_coh * 10.  # NLBW  scaled G
                Elab = E * cm2lab[pin]   # Elab for incoming channel (pair, not ipair)
                Eo = E_scat[ie]*lab2cm if Global else Elab
                Ex[ie] = Eo
                Ry[ie] = x
                Cy[ie] = xa
                Dy[ie] = xc
                if Convolute<=0. and (Global or Elab>0): 
                    print(Eo,x, file=fout)
                    print(Eo,xa,xc, file=fcout)
 
            if Convolute>0.:
                XSEC = XYs.XYs1d(data=(Ex,Ry), dataForm="XsAndYs"  )
                XSEC = XSEC.convolute(conv)
                for ie in range(len(XSEC)):
                    print(XSEC[ie][0],XSEC[ie][1], file=fout)

                XSEC = XYs.XYs1d(data=(Ex,Cy), dataForm="XsAndYs"  )
                XSEC = XSEC.convolute(conv)
                for ie in range(len(XSEC)):
                    print(XSEC[ie][0],XSEC[ie][1], file=fcout)    

                XSEC = XYs.XYs1d(data=(Ex,Dy), dataForm="XsAndYs"  )
                XSEC = XSEC.convolute(conv)
                for ie in range(len(XSEC)):
                    print(XSEC[ie][0],XSEC[ie][1], file=fdout)    

            fout.close()
            fcout.close()

    return
