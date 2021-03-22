
import times
tim = times.times()

import os,math,numpy,cmath,pwd,sys,time,json,re

from CoulCF import cf1,cf2,csigma,Pole_Shifts
from opticals import get_optical_S

from pqu import PQU as PQUModule
from numericalFunctions import angularMomentumCoupling
from xData.series1d  import Legendre

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



def Gomp(gnds,base,emin,emax,jmin,jmax,Dspacing,optical_potentials,hcm,   verbose,debug,inFile,ComputerPrecisions,tim):
        
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
        print(pair,":",kp,' Q =',QI[pair],'R =',prmax[pair])
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
    
    N_opts = max(int( (emax - emin)/Dspacing + 1.5), 0)
    D = (emax-emin)/(N_opts-1) if N_opts > 1 else 0.0
    print("Increase max poles from",n_poles,"to",n_poles + N_opts)
    n_poles += N_opts

    nch = numpy.zeros(n_jsets, dtype=INT)
    npli = numpy.zeros(n_jsets, dtype=INT)
    npl = numpy.zeros(n_jsets, dtype=INT)
    E_poles = numpy.zeros([n_jsets,n_poles], dtype=REAL)
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
        cols = R.nColumns - 1  # ignore energy col
        rows = R.nRows
        E_poles[jset,:rows] = numpy.asarray( R.getColumn('energy','MeV') , dtype=REAL)   # lab MeV
        ncols = cols + 1
        if ReichMoore: 
#             D_poles[jset,:rows] = numpy.asarray( R.getColumn(damping_label,'MeV') , dtype=REAL)   # lab MeV
            for n in range(rows):
                D_poles[jset,n] = R.data[n][damping_pair+1]
            if IFG==1:     D_poles[jset,:] = 2*D_poles[jset,:]**2
        if jmin <= Jpi.spin <= jmax: 
            for ie in range(N_opts):
                e = emin + ie * D
                n = npli[jset]+ie
                E_poles[jset,n] = e
                has_widths[jset,n] = 1
            rows += N_opts
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
        
# CALCULATE OPTICAL-MODEL SCATTERING TO GET PARTIAL WIDTHS

    omfile = open('omp.txt','w')
    sc_info = []
    ncm = int( prmax[ipair] / hcm + 0.5)
    hcm = prmax[ipair]/ncm
    for jset,Jpi in enumerate(RMatrix.spinGroups):
        for c,ch in enumerate(Jpi.channels):
            L =  L_val[jset,c]
            S =  S_val[jset,c]
            for ie in range(N_opts):
                n = npli[jset]+ie
                e = E_poles[jset,n]

                pair = seg_val[jset,c] 
                if pair < 0: continue
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
    
    Smat = get_optical_S(sc_info,ncm)
    SmatMSQ = (Smat * numpy.conjugate(Smat)).real

    AvFormalWidths = Dspacing * (-numpy.log(SmatMSQ)) / (2.*pi)
    for isc,sc in  enumerate(sc_info):
        jset,c,n = sc[:3]
        g_poles[jset,n,c] = AvFormalWidths[isc]
        if IFG==1:  # get rwa
            P =  L_poles[jset,n,c,1]
            g_poles[jset,n,c] = (AvFormalWidths[isc]/(2*P)) ** 0.5
                
# Copy parameters back into GNDS 
    jset = 0
    for Jpi in RMatrix.spinGroups:   # jset values need to be in the same order as before
        parity = '+' if pi_set[jset] > 0 else '-'
#           if True: print('J,pi =',J_set[jset],parity)
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
            row = [ E_poles[jset,n] ]
            if ReichMoore: row.append(0.0)   # ReichMoore damping on new optical poles
            for c in range(cols): 
#               print('Add optical width at j,n,c=',jset,n,c,':',g_poles[jset,n,c])
                row.append(g_poles[jset,n,c])
            R.data.append(row)
        jset += 1
                
    print('\nR-matrix parameters:\n')
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
        
    for jset in range(n_jsets):
        print("  For J/pi = %.1f%s: %i"  % (J_set[jset],'+' if pi_set[jset] > 0 else '-',npl[jset]) )
        for p in range(npl[jset]):
            print('   Pole %3i at Elab = %10.6f (cm %10.6f, obs %10.6f) MeV widths: formal %10.5f, obs %10.5f, damping %10.5f' \
                  % (p,E_poles[jset,p],E_poles[jset,p]/cm2lab[ipair],O_poles[jset,p]/cm2lab[ipair],F_widths[jset,p],O_widths[jset,p],D_poles[jset,p]) )
            if 0.0 < abs(F_widths[jset,p]) < 1e-3: print(68*' ','widths: formal %10.3e, obs %10.3e' % (F_widths[jset,p],O_widths[jset,p]) )
    print()
        
    return
