

##############################################
#                                            #
#    Rflow 0.30      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

import times
tim = times.times()

import os,math,numpy,cmath,pwd,sys,time,json,re

from CoulCF import cf1,cf2,csigma,Pole_Shifts
from write_covariances import write_gnds_covariances
from writeRyaml import write_Ryaml

from pqu import PQU as PQUModule
from numericalFunctions import angularMomentumCoupling
from xData.series1d  import Legendre
from xData import date

from xData.Documentation import documentation as documentationModule
from xData.Documentation import computerCode  as computerCodeModule
# from scipy import interpolate
import fudge.resonances.resolved as resolvedResonanceModule

# CONSTANTS: 
hbc =   197.3269788e0             # hcross * c (MeV.fm)
finec = 137.035999139e0           # 1/alpha (fine-structure constant)
amu   = 931.4940954e0             # 1 amu/c^2 in MeV

coulcn = hbc/finec                # e^2
fmscal = 2e0 * amu / hbc**2
etacns = coulcn * math.sqrt(fmscal) * 0.5e0
pi = 3.1415926536
rsqr4pi = 1.0/(4*pi)**0.5

def now():
    return date.Date( resolution = date.Resolution.time )

def Gflow(gnd,partitions,base,projectile4LabEnergies,data_val,data_p,n_angles,n_angle_integrals,n_totals,n_captures,
        Ein_list, fixedlist, NLMAX,emind,emaxd,pmin,pmax,dmin,dmax,Averaging,Multi,ABES,Grid,nonzero,
        norm_val,norm_info,norm_refs,effect_norm, Lambda,LMatrix,batches,
        init,Search,Iterations,widthWeight,restarts,Background,BG,ReichMoore, 
        Cross_Sections,verbose,debug,inFile,fitStyle,tag,large,ComputerPrecisions,tim):
        
    REAL, CMPLX, INT, realSize = ComputerPrecisions

    PoPs = gnd.PoPs
    projectile = gnd.PoPs[gnd.projectile]
    target     = gnd.PoPs[gnd.target]
    elasticChannel = '%s + %s' % (gnd.projectile,gnd.target)
    if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
    if hasattr(target, 'nucleus'):     target = target.nucleus
    pZ = projectile.charge[0].value; tZ =  target.charge[0].value
    identicalParticles = gnd.projectile == gnd.target
    rStyle = fitStyle.label
    
    rrr = gnd.resonances.resolved
    Rm_Radius = gnd.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')
    bndx = RMatrix.boundaryCondition
    bndv = RMatrix.boundaryConditionValue
    IFG = RMatrix.reducedWidthAmplitudes
    brune = bndx==resolvedResonanceModule.BoundaryCondition.Brune
    if brune: LMatrix = True
#     if brune and not LMatrix:
#         print('Brune basis requires Level-matrix method')
#         LMatrix = True
 
    n_data = data_val.shape[0]
    n_angles0 = 0
    n_angle_integrals0 = n_angles0                                # so [n_angle_integrals0,n_totals0] for angle-integrals
    n_totals0          = n_angle_integrals0 + n_angle_integrals   # so [n_totals0:n_captures0]             for totals
    n_captures0        = n_totals0 + n_totals                     # so [n_totals0:n_data]             for captures
    
    n_jsets = len(RMatrix.spinGroups)
    n_poles = 0
    n_chans = 0
    
    np = len(RMatrix.resonanceReactions)
    ReichMoore = False
    damping = 0
    for pair in range(np):
        print('Partition',pair,'elim,label',RMatrix.resonanceReactions[pair].eliminated,RMatrix.resonanceReactions[pair].label)
        if RMatrix.resonanceReactions[pair].eliminated: 
            ReichMoore = True
            damping = 1
            damping_label = RMatrix.resonanceReactions[pair].label
            print('    Reich-Moore:',damping_label,'\n')
        
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
    cm2lab  = numpy.zeros(np, dtype=REAL)
    pname = ['' for i in range(np)]
    tname = ['' for i in range(np)]

    channels = {}
    pair = 0
    inpair = None
    chargedElastic = False
    print('\nChannels:')
    for partition in RMatrix.resonanceReactions:
        kp = partition.label
        if partition.eliminated: # no two-body kinematics
            partitions[kp] = None
            print('Reaction eliminated:',kp)
            continue
        channels[pair] = kp
        reaction = partition.link.link
        p,t = partition.ejectile,partition.residual
        pname[pair] = p
        tname[pair] = t
        projectile = PoPs[p];
        target     = PoPs[t];

        pMass = projectile.getMass('amu');   tMass =     target.getMass('amu');
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
            prmax[pair] =  partition.getScatteringRadius().getValueAs('fm')
        else:
            prmax[pair] = Rm_global

        if partition.label == elasticChannel:
            ipair = pair  # incoming
        cm2lab[pair] = (pMass + tMass) / tMass

        if p == projectile4LabEnergies and '_e' not in t:  # target must be in gs.
            dlabpair = pair # frame for incoming data_val[:,0]
            
        jp[pair],pt[pair],ep[pair] = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try:
            jt[pair],tt[pair],et[pair] = target.spin[0].float('hbar'), target.parity[0].value, target.energy[0].pqu('MeV').value
        except:
            jt[pair],tt[pair],et[pair] = 0.,1,0.
        print(pair,":",kp,' Q =',QI[pair],'R =',prmax[pair])
        pair += 1
    if verbose: print("\nElastic channel is",elasticChannel,'with IFG=',IFG)
#     if debug: print("Charged-particle elastic:",chargedElastic,",  identical:",identicalParticles,' rStyle:',rStyle)
    npairs  = pair
#     print('npairs:',npairs)
    if not IFG:
        print("Not yet coded for IFG =",IFG)
        sys.exit()
    
#  FIRST: for array sizes:
#
# Find energies in the lab frame of partition 'ipair' as needed for the R-matrix pole energies, not data lab frame:
#
    if dlabpair != ipair: print('Transform main energy vector from',pname[dlabpair],'to',pname[ipair],' projectile lab frames')
    data_val[:,0]  = (data_val[:,0]/cm2lab[dlabpair] - QI[dlabpair] + QI[ipair] ) * cm2lab[ipair]
    E_scat  = data_val[:,0]
#     if dlabpair != ipair: print('Transformed E:',E_scat[:4])
    if debug: print('Energy grid (lab in partition',ipair,'):\n',E_scat)
    Elarge = 0.0
    nExcluded = 0
    if emind is not None: emin = emind
    if emaxd is not None: emax = emaxd
    for i in range(n_data):
        if not max(emin,Elarge) <= E_scat[i] <= emax:
            # print('Datum at energy %10.4f MeV outside evaluation range [%.4f,%.4f]' % (E_scat[i],emin,emax))
#           Elarge = E_scat[i]
            nExcluded += 1
    if nExcluded > 0: print('\n %5i points excluded as outside range [%s, %s]' % (nExcluded,emin,emax))
    
    mu_val = data_val[:,1]
#     print('partitions:',partitions)

    Lmax = 0
    for Jpi in RMatrix.spinGroups:
        R = Jpi.resonanceParameters.table
        n_poles = max(n_poles,R.nRows)
        n = R.nColumns-1
        if ReichMoore: n -= 1
        n_chans = max(n_chans,n)

        partition_channels = numpy.zeros(npairs, dtype=INT)
        for ch in Jpi.channels:
            rr = ch.resonanceReaction
            if RMatrix.resonanceReactions[rr].eliminated: continue
            pair = partitions[rr] #.get(rr,None)
#             print('rr,pair',rr,pair,rr)
            if NLMAX is not None and partition_channels[pair] >= NLMAX: continue
            Lmax = max(Lmax,ch.L)
#             print('       L,Lmax',ch.L,Lmax)
            partition_channels[pair] += 1
    print('Need %i energies in %i Jpi sets with %i poles max, and %i channels max. Lmax=%i' % (n_data,n_jsets,n_poles,n_chans,Lmax))

    nch = numpy.zeros(n_jsets, dtype=INT)
    npl = numpy.zeros(n_jsets, dtype=INT)
    E_poles = numpy.zeros([n_jsets,n_poles], dtype=REAL)
    E_poles_fixed = numpy.zeros([n_jsets,n_poles], dtype=REAL)    # fixed in search
    D_poles = numpy.zeros([n_jsets,n_poles], dtype=REAL)
    D_poles_fixed = numpy.zeros([n_jsets,n_poles], dtype=REAL)    # fixed in search
    has_widths = numpy.zeros([n_jsets,n_poles], dtype=INT)
    g_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=REAL)
    g_poles_fixed = numpy.zeros([n_jsets,n_poles,n_chans], dtype=REAL) # fixed in search
    
    GNDS_order = numpy.zeros([n_jsets,n_poles,n_chans+2], dtype=INT) # [,,0] is energy, 1 is damping, as in GNDS
    GNDS_var   = numpy.zeros([n_jsets,n_poles,n_chans+2], dtype=INT) # [,,0] is energy, 1 is damping, as in GNDS
    J_set = numpy.zeros(n_jsets, dtype=REAL)
    pi_set = numpy.zeros(n_jsets, dtype=INT)
    L_val  =  numpy.zeros([n_jsets,n_chans], dtype=INT) - 1
    S_val  =  numpy.zeros([n_jsets,n_chans], dtype=REAL)
    B_val  =  numpy.zeros([npairs,n_jsets,n_chans], dtype=REAL)
    p_mask =  numpy.zeros([npairs,n_jsets,n_chans], dtype=REAL)
    seg_val=  numpy.zeros([n_jsets,n_chans], dtype=INT) - 1 
    seg_col=  numpy.zeros([n_jsets], dtype=INT) 

    rksq_val  = numpy.zeros([n_data,npairs], dtype=REAL)
    
    eta_val = numpy.zeros([n_data,npairs], dtype=REAL)   # for E>0 only
    
    CF1_val =  numpy.zeros([n_data,np,Lmax+1], dtype=REAL)
    CF2_val =  numpy.zeros([n_data,np,Lmax+1], dtype=CMPLX)
    csigma_v=  numpy.zeros([n_data,np,Lmax+1], dtype=REAL)
    Csig_exp=  numpy.zeros([n_data,np,Lmax+1], dtype=CMPLX)
#     Shift         = numpy.zeros([n_data,n_jsets,n_chans], dtype=REAL)
#     Penetrability = numpy.zeros([n_data,n_jsets,n_chans], dtype=REAL)
    L_diag = numpy.zeros([n_data,n_jsets,n_chans], dtype=CMPLX)
    POm_diag = numpy.zeros([n_data,n_jsets,n_chans], dtype=CMPLX)
    Om2_mat = numpy.zeros([n_data,n_jsets,n_chans,n_chans], dtype=CMPLX)
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
        npl[jset] = rows
        if True: print('J,pi =%5.1f %s, channels %3i, poles %3i' % (J_set[jset],parity,cols,rows) )
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
                if NLMAX is not None and partition_channels[pair] >= NLMAX: continue
                L_val[jset,c] = ch.L
#                 print(' Jset,c =',jset,c,': L=',ch.L)
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
    print('Max channels in each partition:',all_partition_channels,' max=',maxpc,'from NLMAX=',NLMAX)
    for jset in range(n_jsets):
        if debug: print('Channel ranges for each parition:',[[c0[jset,pair],cn[jset,pair]] for pair in range(npairs)])
        
    if debug:
        print('All spins:',All_spins)
        print('All channel spins',Spins)
                
    CSp_diag_in  = numpy.zeros([n_angles,n_jsets,maxpc], dtype=CMPLX)
    CSp_diag_out = numpy.zeros([n_angles,n_jsets,maxpc], dtype=CMPLX)
    
    
#  Calculate Coulomb functions on data Energy Grid
    for pair in range(npairs):
        for ie in range(n_data):
            E = E_scat[ie]/cm2lab[ipair] - QI[ipair] + QI[pair]
            if rmass[pair]!=0:
                k = cmath.sqrt(fmscal * rmass[pair] * E)
            else: # photon!
                k = E/hbc
            if debug: print('ie,E,k = ',ie,E,k)
            rho = k * prmax[pair]
            if abs(rho) <1e-10: 
                print('rho =',rho,'from E,k,r =',E,k,prmax[pair],'from Elab=',E_scat[ie],'at',ie)
            eta  =  etacns * za[pair]*zb[pair] * cmath.sqrt(rmass[pair]/E)
            if E < 0: eta = -eta  #  negative imaginary part for bound states
            PM   = complex(0.,1.); 
            EPS=1e-10; LIMIT = 2000000; ACC8 = 1e-12
            ZL = 0.0
            DL,ERR = cf2(rho,eta,ZL,PM,EPS,LIMIT,ACC8)
            CF2_val[ie,pair,0] = DL
            for L in range(1,Lmax+1):
                RLsq = 1 + (eta/L)**2
                SL   = L/rho + eta/L
                CF2_val[ie,pair,L] = RLsq/( SL - CF2_val[ie,pair,L-1]) - SL

            if E > 0.:
                CF1_val[ie,pair,Lmax] = cf1(rho.real,eta.real,Lmax,EPS,LIMIT)
                for L in range(Lmax,0,-1):
                    RLsq = 1 + (eta.real/L)**2
                    SL   = L/rho.real + eta.real/L
                    CF1_val[ie,pair,L-1] = SL - RLsq/( SL + CF1_val[ie,pair,L]) 

            CF1_val[ie,pair,:] *=  rho.real
            CF2_val[ie,pair,:] *=  rho
            rksq_val[ie,pair] = 1./max(abs(k)**2, 1e-20) 
            
            if E > 0.:
                eta_val[ie,pair] = eta.real
                csigma_v[ie,pair,:] = csigma(Lmax,eta)
                for L in range(Lmax+1):
                    Csig_exp[ie,pair,L] = cmath.exp(complex(0.,csigma_v[ie,pair,L]-csigma_v[ie,pair,0]))
            else:
                eta_val[ie,pair] = 0.0
                Csig_exp[ie,pair,:] = 1.0
            
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
                D_poles[jset,n] = R.data[n][damping]
            if IFG==1:     D_poles[jset,:] = 2*D_poles[jset,:]**2
        for n in range(rows):
            for c in range(ncols):
                GNDS_order[jset,n,c] = G_order  # order of variables in GNDS and ENDF, needed for covariance matrix
                G_order += 1 

        widths = [R.getColumn( col.name, 'MeV**(1/2)' ) for col in R.columns if col.name != 'energy']
        
#         if verbose:  print("\n".join(R.toXMLList()))       
        n = None
        c = 0
        for pair in range(npairs):
            cp = 0
            for ch in Jpi.channels:
                if NLMAX is not None and cp >= NLMAX: continue

                ic = c - c0[jset,pair]
                rr = ch.resonanceReaction
                pairc = partitions.get(rr,None)
                if pairc is None or pairc!=pair: continue
                m = ch.columnIndex - 1
                g_poles[jset,:rows,c] = numpy.asarray(widths[m][:],  dtype=REAL) 
                
                if nonzero  is not None:
                    for n  in range(rows):
                        if abs(g_poles[jset,n,c]) < 1e-20:
                            g_poles[jset, n, c] = nonzero

            # Find S, P, phi, sigma for all data points
                for ie in range(n_data):
                    pin = data_p[ie,0]
                    pout= data_p[ie,1]
                    if pout == -1: pout = pin # to get total cross-section
                    #  pout == -2 for capture cross-section
                    B = None
                    if bndx == resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum:
                        B = -ch.L
                    elif bndx == resolvedResonanceModule.BoundaryCondition.Brune:
                        B = None
                    elif bndx == resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction:
                        B = None
                    elif bndx == resolvedResonanceModule.BoundaryCondition.Given:
                        B = float(bndv)
                    else:
                        print('Boundary condition',bndx,'and value',bndv,'not recognized')
                        sys.exit()
                    if ch.boundaryConditionValue is not None:
                        B = float(ch.boundaryConditionValue)
                    
                    if B is not None: B_val[pair,jset,c] = B

                    DL = CF2_val[ie,pair,ch.L]
                    S = DL.real
                    P = DL.imag
                    F = CF1_val[ie,pair,ch.L]
                    Psr = math.sqrt(abs(P))
                    phi = - math.atan2(P, F - S)
                    Omega = cmath.exp(complex(0,phi))
                    if bndx == resolvedResonanceModule.BoundaryCondition.Brune:
                        L_diag[ie,jset,c]       = DL
                    elif B is None:
                        L_diag[ie,jset,c]       = complex(0.,P)
                    else:
                        L_diag[ie,jset,c]       = DL - B

                    POm_diag[ie,jset,c]      = Psr * Omega
                    Om2_mat[ie,jset,c,c]     = Omega**2
#                     CS_diag[ie,jset,c]       = Csig_exp[ie,pair,ch.L]
                    if ie < n_angles:
                        if pair==pin : CSp_diag_in[ie,jset,ic]       = Csig_exp[ie,pair,ch.L]
                        if pair==pout: CSp_diag_out[ie,jset,ic]      = Csig_exp[ie,pair,ch.L]
                c += 1
                cp += 1
        if debug:
            print('J set %i: E_poles \n' % jset,E_poles[jset,:])
            print('g_poles \n',g_poles[jset,:,:])
        jset += 1   

    if brune:  # L_poles: Shift+iP functions at pole positions for Brune basis   
        if Grid == 0.0:
            L_poles = numpy.zeros([n_jsets,n_poles,n_chans,2], dtype=REAL)
            dLdE_poles = numpy.zeros([n_jsets,n_poles,n_chans,2], dtype=REAL)
    #         EO_poles =  numpy.zeros([n_jsets,n_poles], dtype=REAL) 
            EO_poles = E_poles.copy()
            Pole_Shifts(L_poles,dLdE_poles, EO_poles,has_widths, seg_val,1./cm2lab[ipair],QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
            
        else : # make a linear grid of Shift functions for use for each pole
            Lowest_pole_energy = numpy.amin(E_poles) - 10.
            Highest_pole_energy = numpy.amax(E_poles) + 20.
            N_gridE = int( (Highest_pole_energy - Lowest_pole_energy) / Grid ) + 1
            print('Make grid of S+iP at %s MeV spacings for e.g. Brune level matrix with %i points from Elab from %.3f to %.3f' % (Grid,N_gridE,Lowest_pole_energy,Highest_pole_energy))
#             GRsize = N_gridE*n_jsets*n_chans*2*realSize / 1e9
#             if GRsize > 0.01: print('    Grid storage size = %6.3f GB' % GRsize)
            L_poles = numpy.zeros([N_gridE,n_jsets,n_chans,2], dtype=REAL)
            LGB = N_gridE*n_jsets*n_chans*2 * realSize / 1e9
            if LGB>0.01: print('    Grid storage size = %.3f GB' % LGB)
            dLdE_poles = None
            EO_poles = None
            CF2_L = numpy.zeros(Lmax+1, dtype=CMPLX)
            
            Egrid = numpy.zeros(N_gridE)
            for ie in range(N_gridE):
                Egrid[ie] = Lowest_pole_energy + ie*Grid   # ELab on the GNDS projectile frame (ipair) 

            for pair in range(npairs):
                for ie in range(N_gridE):
                    Escat = Egrid[ie]
                    E = Escat/cm2lab[ipair] - QI[ipair] + QI[pair]  # Ecm in this partition 'pair'
                    if rmass[pair]!=0:
                        k = cmath.sqrt(fmscal * rmass[pair] * E)
                    else: # photon!
                        k = E/hbc
                    if debug: print('ie,E,k = ',ie,E,k)
                    rho = k * prmax[pair]
                    if debug and abs(rho) <1e-10: 
                        print('rho =',rho,'from E,k,r =',E,k,prmax[pair],'from Elab=',E_scat[ie],'at',ie)
                    if abs(E) <1e-5: continue
                    eta  =  etacns * za[pair]*zb[pair] * cmath.sqrt(rmass[pair]/E)
                    if E < 0: eta = -eta  #  negative imaginary part for bound states
                    c_E = prmax[pair] * math.sqrt(fmscal*rmass[pair]) 
                    PM   = complex(0.,1.); 
                    EPS=1e-8; LIMIT = 2000000; ACC8 = 1e-12
                    ZL = 0.0
                    CF,ERR = cf2(rho,eta,ZL,PM,EPS,LIMIT,ACC8) 
                    CF2_L[0] = CF 
                    for L in range(1,Lmax+1):
                        RLsq = 1 + (eta/L)**2
                        SL   = L/rho + eta/L
                        CF2_L[L] = RLsq/( SL - CF2_L[L-1]) - SL
                    for jset in range(n_jsets):
                        for c in range(nch[jset]):
                            if seg_val[jset,c] != pair: continue
                            DL = CF2_L[ L_val[jset,c] ] * rho
                            if bndx == resolvedResonanceModule.BoundaryCondition.Brune:
                                pass
                            elif B is None:
                                DL       = complex(0.,DL.imag)   # 'S' or 
                            else:
                                DL       = DL - B_val[pair,jset,c]
                            L_poles[ie,jset,c,:] = (DL.real,DL.imag)

#             for jset in range(n_jsets):
#                 for c in range(nch[jset]):
#                     for ie in range(N_gridE):
#                         if ie % 1000 == 1: print(jset,c,'%10.6f' % Egrid[ie],'L_poles[ie,jset,c,:]:',L_poles[ie,jset,c,:]) #,(L_poles[ie,jset,c,0]-L_poles[ie-1,jset,c,0])/Grid )
#
    else:
        L_poles = None
        dLdE_poles = None
        EO_poles = None

#     print('E_poles \n',E_poles[:,:])
#     print('D_poles \n',D_poles[:,:])
#     print('g_poles \n',g_poles[:,:,:])
#     print('norm_val \n',norm_val[:])
    if Search:
        print('\n Search:',Search,'to',Iterations,'iterations')

    n_norms = norm_val.shape[0]
    fixed_norms= numpy.zeros([n_norms], dtype=REAL) # fixed in search
    t_vars = n_poles* n_jsets + n_poles*n_jsets*n_chans + n_norms   # max possible # variables
    fixedpars = numpy.zeros(t_vars, dtype=REAL)
    fixedloc  = numpy.zeros([t_vars,1], dtype=INT)  
    GNDS_loc  = numpy.zeros([t_vars,1], dtype=INT)  
    NORM_var =numpy.zeros([t_vars]) # searchpars_n[border[2]:border[3]] ** 2

    searchnames = []
    fixednames = []
    search_vars = []
    POLE_details = {}

    ip = 0
    ifixed = 0
    border = numpy.zeros(5, dtype=INT)     # variable parameters
    frontier = numpy.zeros(5, dtype=INT)   # fixed parameters
    patterns = [ re.compile(fix_regex) for fix_regex in fixedlist] 
    fixedlistex = set()

    border[0] = 0;  frontier[0] = 0
    for jset in range(n_jsets):
        parity = '+' if pi_set[jset] > 0 else '-'
        for n in range(n_poles):
            i = jset*n_poles+n
            E = E_poles[jset,n]
            Ecm = E/cm2lab[ipair]
            if E == 0: continue   # invalid energy: filler
            nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E)
            varying = abs(Ecm) < Background  and n < npl[jset]
            if pmin is not None and pmax is not None and pmin > pmax: 
                varying = varying and (E > pmin or E < pmax)
            else:
                if pmin is not None: varying = varying and Ecm > pmin
                if pmax is not None: varying = varying and Ecm < pmax
            for pattern in patterns:
                 varying = varying and not pattern.match(nam) 
#             print('Pole',jset,n,'named',nam,'at',E, 'vary:',varying)
            if varying: 
#                 searchparms[ip] = E
#                 searchloc[ip,0] = i
                searchnames += [nam]
                search_vars.append([E,i])
                GNDS_loc[ip] = GNDS_order[jset,n,0]
                GNDS_var[jset,n,0] = 1
                det = ('E',jset,n,ip,float(J_set[jset]),int(pi_set[jset]))
#                 print('det:',det,type(det))
                POLE_details[ip] =  det
                ip += 1
            else:
                fixedlistex.add(nam)
                E_poles_fixed[jset,n] = E_poles[jset,n]
                if Search:
                    print('    Fixed %5.1f%1s pole %2i at E = %7.3f MeV(lab), %7.3f MeV(cm)' % (J_set[jset],parity,n,E,E/cm2lab[ipair]) )
                if nam not in fixedlistex and BG:
                    nam='BG:%.1f%s' % (J_set[jset],parity)
                # print('E[',jset,n,'] is fixed',ifixed,'at',E_poles[jset,n])
                fixedpars[ifixed] = E_poles[jset,n]
                fixedloc[ifixed,0] = i
                fixednames += [nam]
                ifixed += 1
    border[1] = ip
    frontier[1] = ifixed
    if border[1]>0 and brune and Grid == 0.0 and Search:
        if not ABES:
            print('Stop. You request fitting of Brune energies, but method not accurate. ABES not set')
            sys.exit()
        else:
            print('ABES set to Allow Brune Energy Shifts')


    for jset in range(n_jsets):
        parity = '+' if pi_set[jset] > 0 else '-'
        for n in range(n_poles):
            E = E_poles[jset,n]
            Ecm = E/cm2lab[ipair] + QI[ipair]
            for c in range(n_chans):
                if L_val[jset,c] < 0: continue   # invalid partial wave: blank filler
                if E == 0: continue   # invalid energy: filler
                i = (jset*n_poles+n)*n_chans+c
                if abs(E) < Background or not BG:
                    nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E)
                else:
                    nam='BG:%.1f%s' % (J_set[jset],parity)
                wnam = 'w'+str(c)+','+ (nam[1:] if nam[0]=='P' else nam)
#                 varying = abs(g_poles[jset,n,c])>1e-20 
                varying = c < nch[jset] and n < npl[jset] 
                if pmin is not None and pmax is not None and pmin > pmax:   # -p,-P fix both energies and widths
                    varying = varying and (Ecm > pmin or Ecm < pmax)
                else:
                    if pmin is not None: varying = varying and Ecm > pmin
                    if pmax is not None: varying = varying and Ecm < pmax

                for pattern in patterns:
                    matching = pattern.match(wnam)
                    varying = varying and not pattern.match(wnam)   
#                     print('     varying=',varying,'after',wnam,'matches',pattern.match(wnam),matching,True if matching else False,pattern)              
                
#                 print('Width',jset,n,c,'named',wnam,'from',nam,E, 'vary:',varying,'\n')
                if varying:
#                     searchparms[ip] = g_poles[jset,n,c]
#                     searchloc[ip,0] = i
                    searchnames += [wnam]
                    search_vars.append([g_poles[jset,n,c],i])
                    GNDS_loc[ip] = GNDS_order[jset,n,c+1]
                    GNDS_var[jset,n,c+1] = 1
                    det = ('W',jset,n,ip,seg_val[jset,c],L_val[jset,c],S_val[jset,c])
                    POLE_details[ip] = det
                    ip += 1
                else:   # fixed
                    fixedlistex.add(wnam)
                    # print('g[',jset,n,c,'] is fixed',ifixed,'at',g_poles[jset,n,c])
                    fixedpars[ifixed] = g_poles[jset,n,c]
                    g_poles_fixed[jset,n,c] = g_poles[jset,n,c]
                    fixedloc[ifixed,0] = i
                    fixednames += [wnam]
                    ifixed += 1                    
    border[2] = ip
    frontier[2] = ifixed
    
    for ni in range(n_norms):
        nnam = norm_refs[ni][0]
        snorm = math.sqrt(norm_val[ni])
        varying = norm_info[ni,2] < 1
        for pattern in patterns:
             varying = varying and not pattern.match(nnam)
        if varying:
#             searchparms[ip] = snorm
#             searchloc[ip,0] = ni
            searchnames += [nnam]
            search_vars.append([snorm,ni])
            det = ('N',ni,ip)
#           print('det:',det)
            POLE_details[ip] =  det
            NORM_var[ni] = ip
            ip += 1
        else:
            fixedlistex.add(nnam)
            fixedpars[ifixed] =  snorm
            fixed_norms[ni] = snorm
            fixedloc[ifixed,0] = ni
            fixednames += [nnam]
            ifixed += 1   
    border[3] = ip
    frontier[3] = ifixed
    
    if ReichMoore:
        for jset in range(n_jsets):
            parity = '+' if pi_set[jset] > 0 else '-'
            for n in range(n_poles):
                i = jset*n_poles+n
                E = E_poles[jset,n]
                Ecm = E/cm2lab[ipair]
                if E == 0: continue   # invalid energy: filler
                D = D_poles[jset,n]
                sD = math.sqrt(D)
                if D == 0: continue   # decided to vary only non-zero damping
                nam='PJ%.1f%s:D%.3f' % (J_set[jset],parity, E)
                varying =  n < npl[jset]
                if dmin is not None and dmax is not None and dmin > dmax: 
                    varying = varying and (E > dmin or E < dmax)
                else:
                    if dmin is not None: varying = varying and Ecm > dmin
                    if dmax is not None: varying = varying and Ecm < dmax
                for pattern in patterns:
                     varying = varying and not pattern.match(nam) 
#               print('Pole',jset,n,'named',nam,'at',E, 'D varies:',varying)
                if varying: 
                    searchnames += [nam]
                    search_vars.append([sD,i])
                    GNDS_loc[ip] = GNDS_order[jset,n,0]
                    ip += 1
                else:
                    fixedlistex.add(nam)
                    D_poles_fixed[jset,n] = sD
                    if Search:
                        print('    Fixed %5.1f%1s pole %2i  D = %.3e MeV at E = %7.3f' % (J_set[jset],parity,n,D,E) )
                    # print('E[',jset,n,'] is fixed',ifixed,'at',E_poles[jset,n])
                    fixedpars[ifixed] = sD
                    fixedloc[ifixed,0] = i
                    fixednames += [nam]
                    ifixed += 1
    border[4] = ip
    frontier[4] = ifixed 
    
    n_pars = border[4]
    n_fixed = frontier[4]
    
    print('Variable borders:',border,'and Fixed frontiers:',frontier)
    searchparms = numpy.zeros(n_pars, dtype=REAL)
    searchloc  = numpy.zeros([n_pars,1], dtype=INT)   
    for ip in range(n_pars):
        searchparms[ip] = search_vars[ip][0]
        searchloc[ip,0] = search_vars[ip][1]

    print('Variable parameters - E,w,norms,D: ',border[1]-border[0],border[2]-border[1],border[3]-border[2],border[4]-border[3],' =',n_pars) 
    print('Fixed    parameters - E,w,norms,D: ',frontier[1]-frontier[0],frontier[2]-frontier[1],frontier[3]-frontier[2],frontier[4]-frontier[3],' =',n_fixed) 
    print('# zero widths  =',numpy.count_nonzero(g_poles == 0) ,'\n')
    n_dof = n_data - border[4]
    
    if debug:
        print('\n Variable parameters:',' '.join(searchnames)) 
        print('Fixed    parameterlist:',' '.join(fixedlistex))
    print('Searching on pole energies:',searchnames[border[0]:border[1]])
    print('Keep fixed   pole energies:',fixednames[frontier[0]:frontier[1]])
    print('Searching on widths :',searchnames[border[1]:border[2]])
    print('Keep fixed   widths:',fixednames[frontier[1]:frontier[2]])
    print('Searching on norms :',searchnames[border[2]:border[3]])
    print('Keep fixed   norms:',fixednames[frontier[2]:frontier[3]])
    print('Searching on damping widths: [',' '.join(['%.2e' % d**2 for d in searchparms[border[3]:border[4]]]),']') 
    print('L4 norm of widths:',numpy.sum(searchparms[border[1]:border[2]]**4))
    
    if brune and False:
        for jset in range(n_jsets):
            for n in range(npl[jset]):
                print('j/n=',jset,n,' E_pole: %10.6f' % EO_poles[jset,n])
                for c in range(nch[jset]):
                     print("      S, S' %10.6f, %10.6f" % (L_poles[jset,n,c].real,dLdE_poles[jset,n,c].real))
                                 
              
## ANGULAR-MOMENTUM ARRAYS:

    gfac = numpy.zeros([n_data,n_jsets], dtype=REAL)
    for ie in range(n_data):
        pin = data_p[ie,0]   # incoming partition
        for jset in range(n_jsets):
            denom = (2.*jp[pin]+1.) * (2.*jt[pin]+1)
            gfac[ie,jset] = pi * (2*J_set[jset]+1) * rksq_val[ie,pin] / denom * 10.  # mb
       
    Gfacc = numpy.zeros(n_angles, dtype=REAL)    
    NL = 2*Lmax + 1
    Pleg = numpy.zeros([n_angles,NL], dtype=REAL)
#     ExptAint = numpy.zeros([n_angle_integrals,npairs, npairs], dtype=REAL)
#     ExptTot = numpy.zeros([n_totals,npairs], dtype=REAL)

    for ie in range(n_angles):
        pin = data_p[ie,0]
        jproj = jp[pin]
        jtarg = jt[pin]
        denom = (2.*jproj+1.) * (2.*jtarg+1)
        Gfacc[ie]    = pi * rksq_val[ie,pin] / denom  * 10.   # mb.  Likd gfac, but no (2J+1) factor
        mu = mu_val[ie]
        if abs(mu)>1.: 
            print('\nData pt ',ie,data_p[ie,:],'has bad mu:',mu_val[ie])
            print(data_val[ie,:])
            print(data_p[ie,:])
            continue
            sys.exit()
        for L in range(NL):
            Pleg[ie,L] = Legendre(L, mu)
        
    if chargedElastic:
        Rutherford = numpy.zeros([n_angles], dtype=REAL)
        InterferenceAmpl = numpy.zeros([n_angles, n_jsets, maxpc], dtype=CMPLX)
        
        for ie in range(n_angles):
            pin = data_p[ie,0]
            pout= data_p[ie,1]
            if pin==pout:
                mu = mu_val[ie]
                if mu>1.:
                    print("Error: mu",mu)
                shthsq = (1-mu) * 0.5
                jproj = jp[pin]
                jtarg = jt[pin]
                denom = (2.*jproj+1.) * (2.*jtarg+1)
                eta = eta_val[ie,pin].real
                if eta !=0:
                    Coulmod  = eta.real * rsqr4pi / shthsq
                    CoulAmpl = Coulmod * cmath.exp(complex(0., - eta*math.log(shthsq) ))
                else:
                    Coulmod = 0.0
                    CoulAmpl = 0.0
                Rutherford[ie] = denom * Coulmod**2
            
                for jset in range(n_jsets):
                    J = J_set[jset]
                    for c in range(n_chans):
                        ic = c - c0[jset,pin]
                        if seg_val[jset,c] == pin:
                            L = L_val[jset,c]
                            if L < 0: continue
                            InterferenceAmpl[ie,jset,ic] = (2*J+1) * Pleg[ie,L] * 2 * rsqr4pi * CoulAmpl.conjugate()
    else:
        Rutherford, InterferenceAmpl = None, None

    
    NS = len(All_spins)
    ZZbar = numpy.zeros([NL,NS,n_jsets,n_chans,n_jsets,n_chans], dtype=REAL)

    def n2(x): return(int(2*x + 0.5))
    def i2(i): return(2*i)
    def triangle(x,y,z): return (  abs(x-y) <= z <= x+y )

    for iS,S in enumerate(All_spins):
        for jset1 in range(n_jsets):
            J1 = J_set[jset1]
            for c1 in range(n_chans):
                L1 = L_val[jset1,c1]
                if not triangle( L1, S, J1) : continue
                for jset2 in range(n_jsets):
                    J2 = J_set[jset2]
                    for c2 in range(n_chans):
                        L2 = L_val[jset2,c2]
                        if not triangle( L2, S, J2) : continue
                        for L in range(NL):                    
                            ZZbar[L,iS,jset2,c2,jset1,c1] = angularMomentumCoupling.zbar_coefficient(i2(L1),n2(J1),i2(L2),n2(J2),n2(S),i2(L))

    BB = numpy.zeros([n_data,NL], dtype=REAL)
    
    if n_angles > 0:
        cc = (n_jsets*n_chans**2)**2
        ccp = (n_jsets*maxpc**2)**2
#         print('AAL, AA sizes= %5.3f, %5.3f GB' % (cc*npairs**2*NL*realSize/1e9, cc*n_angles*realSize/1e9 ),'from %s*(%s*%s^2)^2 reals' % (n_angles,n_jsets,n_chans))
        print('AAL, AA sizes/p= %5.3f, %5.3f GB' % (ccp*npairs**2*NL*realSize/1e9, ccp*n_angles*realSize/1e9 ),'from %s*(%s*%s^2)^2 reals' % (n_angles,n_jsets,maxpc))
        AAL = numpy.zeros([npairs,npairs, n_jsets,maxpc,maxpc, n_jsets,maxpc,maxpc ,NL], dtype=REAL)

        for rr_in in RMatrix.resonanceReactions:
            if rr_in.eliminated: continue
            inpair = partitions[rr_in.label]

            for rr_out in RMatrix.resonanceReactions:
                if rr_out.eliminated: continue
                pair = partitions[rr_out.label]
                
                for S_out in Spins[pair]:
                    for S_in in Spins[inpair]:
    #                     print('>> S_in:',S_in)
                        for iS,S in enumerate(All_spins):
                            for iSo,So in enumerate(All_spins):
                                if abs(S-S_in)>0.1 or abs(So-S_out)>0.1: continue
                                phase = (-1)**int(So-S) / 4.0


                                for jset1 in range(n_jsets):
                                    J1 = J_set[jset1]
                                    for c1 in range(n_chans):
                                        if seg_val[jset1,c1] != inpair: continue
                                        if abs(S_val[jset1,c1]-S) > 0.1 : continue
                                        ic1 = c1 - c0[jset1,inpair]

                                        for c1_out in range(n_chans):
                                            if seg_val[jset1,c1_out] != pair: continue
                                            if abs(S_val[jset1,c1_out]-So) > 0.1 : continue
                                            ic1_out = c1_out - c0[jset1,pair]

                                            for jset2 in range(n_jsets):
                                                J2 = J_set[jset2]
                                                for c2 in range(n_chans):
                                                    if seg_val[jset2,c2] != inpair: continue
                                                    if abs(S_val[jset2,c2]-S) > 0.1 : continue
                                                    ic2 = c2 - c0[jset2,inpair]

                                                    for c2_out in range(n_chans):
                                                        if seg_val[jset2,c2_out] != pair: continue
                                                        if abs(S_val[jset2,c2_out]-So) > 0.1 : continue
                                                        ic2_out = c2_out - c0[jset2,pair]
        
                                                        for L in range(NL):
                                                            ZZ = ZZbar[L,iS,jset2,c2,jset1,c1] * ZZbar[L,iSo,jset2,c2_out,jset1,c1_out] 
                                                            AAL[pair,inpair, jset2,ic2_out,ic2, jset1,ic1_out,ic1,L] += phase * ZZ / pi 

    #     AA = numpy.zeros([n_angles, n_jsets,maxpc,maxpc, n_jsets,maxpc,maxpc  ], dtype=REAL)

    #     for ie in range(n_angles):
    #         pin = data_p[ie,0]
    #         pout= data_p[ie,1]
    #         for L in range(NL):
    #             AA[ie, :,:,:, :,:,:] += AAL[pin,pout, :,:,:, :,:,:, L] * Pleg[ie,L]
    else:
        AAL = None

    E_poles_fixed_v = numpy.ravel(E_poles_fixed)
    g_poles_fixed_v = numpy.ravel(g_poles_fixed)
    D_poles_fixed_v = numpy.ravel(D_poles_fixed)

    n_angles0 = 0
    n_angle_integrals0 = n_angles0 + n_angles                     # so [n_angle_integrals0,n_totals0] for angle-integrals
    n_totals0          = n_angle_integrals0 + n_angle_integrals   # so [n_totals0:n_captures0]             for totals
    n_captures0        = n_totals0 + n_totals                     # so [n_totals0:n_data]             for captures

    searchpars0 = searchparms
    n_pars = searchpars0.shape[0]
    print('Number of search parameters:',n_pars)
    
    if init is not None:
        ifile = open(init[1],'r')
        for i in range(1,int(init[0])):
            vals = ifile.readline()
        vals = ifile.readline().replace('[','').replace(']','').split()
        print('\nRestart at chisq/pt',vals[0],'and data chisq/pt',vals[2],'\n')
        searchpars0 = numpy.asarray([float(v) for v in vals[3:]], dtype=REAL)
        if n_pars != len(searchpars0):
            print('Number of reread search parameters',len(searchpars0),' is not',n_pars,'now expected. STOP')
            sys.exit()
#         if nonzero  is not None:
#             for n  in range(rows):
#                 if abs(g_poles[jset,n,c]) < 1e-20:
#                     g_poles[jset, n, c] = nonzero
                            
    if Cross_Sections:
        
        ExptAint = numpy.zeros([n_angle_integrals,npairs, npairs], dtype=REAL)
        ExptTot = numpy.zeros([n_totals,npairs], dtype=REAL)
        ExptCap = numpy.zeros([n_captures,npairs], dtype=REAL)
        for ie in range(n_angle_integrals):
            pin = data_p[n_angle_integrals0+ie,0]
            pout= data_p[n_angle_integrals0+ie,1]
            ExptAint[ie,pout,pin] = 1.
        for ie in range(n_totals):
            pin = data_p[n_totals0+ie,0]
            ExptTot[ie,pin] = 1.
        for ie in range(n_captures):
            pin = data_p[n_captures0+ie,0]
            ExptCap[ie,pin] = 1.
                        
        CS_diag = numpy.zeros([n_data,n_jsets,n_chans], dtype=CMPLX)
        for jset in range(n_jsets):
            for c in range(n_chans):
                pair = seg_val[jset,c]
                if pair >= 0: CS_diag[:,jset,c] = Csig_exp[:,pair,L_val[jset,c]]
                
        # for all cross-sections
        gfac_s = numpy.zeros([n_data,n_jsets,npairs,maxpc], dtype=REAL)
        for jset in range(n_jsets):
            for pair in range(npairs):     # incoming partition
                denom = (2.*jp[pair]+1.) * (2.*jt[pair]+1)
                nic = cn[jset,pair] - c0[jset,pair]
                for ie in range(n_data):
                    gfac_s[ie,jset,pair,0:nic] = pi * (2*J_set[jset]+1) * rksq_val[ie,pair] / denom * 10.  # mb
    else:
        ExptAint,ExptTot,ExptCap,CS_diag,p_mask,gfac_s = None, None, None, None, None, None


            
    print("To start tf: ",tim.toString( ))
    
    EBU = dmin*cm2lab[ipair] if dmin is not None else 0.
    if Lambda is not None:
         rule = '(E - %.3f)^%5.3f' % (EBU,Lambda,EBU) if Lambda > 0 else '1 - exp(-%f*(E-%.3f))' % (-Lambda,EBU)
         print('\nModulate damping widths by factor %s, or 0.0 for E < %.3f (lab energies)\n' % (rule,EBU) )
    sys.stdout.flush()

################################################################    
## TENSORFLOW CALL:

#     ComputerPrecisions = (REAL, CMPLX, INT, realsize)

    Channels = [ipair,nch,npl,pname,tname,za,zb,QI,cm2lab,rmass,prmax,L_val,c0,cn,seg_val]
    CoulombFunctions_data = [L_diag, Om2_mat,POm_diag,CSp_diag_in,CSp_diag_out, Rutherford, InterferenceAmpl, Gfacc,gfac]    # batch n_data
   
    if brune: 
        if Grid == 0.0 :
            CoulombFunctions_poles = [L_poles,dLdE_poles,EO_poles,has_widths]                  # batch n_jsets
        else:
            CoulombFunctions_poles = [L_poles,Lowest_pole_energy,Highest_pole_energy]      # L = S+iP on a regular grid
    else:
        CoulombFunctions_poles = [None,None,None] 

    Dimensions = [n_data,npairs,n_jsets,n_poles,n_chans,n_angles,n_angle_integrals,n_totals,n_captures,NL,maxpc,batches]
    Logicals = [LMatrix,brune,Grid,Lambda,EBU,chargedElastic, debug,verbose]
    Search_Control = [searchloc,border,E_poles_fixed_v,g_poles_fixed_v,D_poles_fixed_v, fixed_norms,norm_info,effect_norm,data_p, AAL,base, Search,Iterations,Averaging,widthWeight,restarts,Cross_Sections]

    Data_Control = [Pleg, ExptAint,ExptTot,ExptCap,CS_diag,p_mask,gfac_s]     # Pleg + extra for Cross-sections  
    
#     print('Channels:',Channels)    
#     print('CoulombFunctions_data:',CoulombFunctions_data)    
#     print('CoulombFunctions_poles:',CoulombFunctions_poles)
#     print('Dimensions:',Dimensions)
#     print('Logicals:',Logicals)
#     print('Search_Control:',Search_Control)
#     print('other args:',Multi,ComputerPrecisions,searchpars0,data_val)

    from evaluate import evaluate
    searchpars_n, chisq_n, grad1, inverse_hessian, chisq0_n,grad0, A_tF_n, XS_totals = evaluate(Multi,ComputerPrecisions, Channels,
        CoulombFunctions_data,CoulombFunctions_poles, Dimensions,Logicals, 
        Search_Control,Data_Control, searchpars0, data_val, tim)


    ww = numpy.sum(searchpars_n[border[1]:border[2]]**4) * widthWeight
    print("Finished tf: ",tim.toString( ))
    sys.stdout.flush()
#  END OF TENSORFLOW CALL
################################################################
                    
    if Search:     
#  Write back fitted parameters into evaluation:
        E_poles = numpy.zeros([n_jsets,n_poles], dtype=REAL) 
        g_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=REAL)
        norm_val =numpy.zeros([n_norms]) # searchpars_n[border[2]:border[3]] ** 2
        
        chisqpdof = chisq_n/n_dof

        newname = {}
        for ip in range(border[0],border[1]): #### Extract parameters after previous search:
            i = searchloc[ip,0]
            jset = i//n_poles;  n = i%n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            E_poles[jset,n] = searchpars_n[ip]
            varying = abs(E_poles[jset,n]) < Background and searchnames[ip] not in fixedlistex
            nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  searchnames[ip] not in fixedlistex and BG: nam = 'BG:%.1f%s' % (J_set[jset],parity)
#             print(ip,'j,n E',searchnames[ip],'renamed to',nam)
            newname[searchnames[ip]] = nam

        for ip in range(frontier[0],frontier[1]): #### Extract parameters after previous search:
            i = fixedloc[ip,0]
            jset = i//n_poles;  n = i%n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            E_poles[jset,n] = fixedpars[ip]
            varying = abs(E_poles[jset,n]) < Background and  fixednames[ip] not in fixedlistex
            nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  fixednames[ip] not in fixedlistex and BG: nam = 'BG:%.1f%s' % (J_set[jset],parity)
#             print(ip,'j,n fixed E',fixednames[ip],'renamed to',nam)
            newname[fixednames[ip]] = nam        
                    
        for ip in range(border[1],border[2]): ##                i = (jset*n_poles+n)*n_chans+c
            i = searchloc[ip,0]
            c = i%n_chans;  n = ( (i-c)//n_chans )%n_poles; jset = ((i-c)//n_chans -n)//n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            g_poles[jset,n,c] = searchpars_n[ip]
            nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  searchnames[ip] not in fixedlistex and BG: nam = 'BG:%.1f%s' % (J_set[jset],parity)
            wnam = 'w'+str(c)+','+ (nam[1:] if nam[0]=='P' else nam)
#             print(ip,'j,n,c width',searchnames[ip],'renamed to',wnam)
            newname[searchnames[ip]] = wnam        
        
        for ip in range(frontier[1],frontier[2]): ##                i = (jset*n_poles+n)*n_chans+c
            i = fixedloc[ip,0]
            c = i%n_chans;  n = ( (i-c)//n_chans )%n_poles; jset = ((i-c)//n_chans -n)//n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            g_poles[jset,n,c] = fixedpars[ip]
            nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  fixednames[ip] not in fixedlistex and BG: nam = 'BG:%.1f%s' % (J_set[jset],parity)
            wnam = 'w'+str(c)+','+ (nam[1:] if nam[0]=='P' else nam)
#             print(ip,'j,n,c fixed width',fixednames[ip],'renamed to',wnam)
            newname[fixednames[ip]] = wnam        
#         print('newname:',newname)

        for ip in range(border[2],border[3]): ## merge variable norms
            ni = searchloc[ip,0]
            norm_val[ni] = searchpars_n[ip]**2
        for ip in range(frontier[2],frontier[3]): ## merge fixed norms
            ni = fixedloc[ip,0]
            norm_val[ni] = fixed_norms[ni]**2

        for ip in range(border[3],border[4]): #### Extract parameters after previous search:
            i = searchloc[ip,0]
            jset = i//n_poles;  n = i%n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            D_poles[jset,n] = searchpars_n[ip]**2
            varying = abs(E_poles[jset,n]) < Background and searchnames[ip] not in fixedlistex
            nam='PJ%.1f%s:D%.3f' % (J_set[jset],parity, D_poles[jset,n])
            newname[searchnames[ip]] = nam

        for ip in range(frontier[3],frontier[4]): #### Extract parameters after previous search:
            i = fixedloc[ip,0]
            jset = i//n_poles;  n = i%n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            D_poles[jset,n] = fixedpars[ip]**2
            varying = abs(E_poles[jset,n]) < Background and  fixednames[ip] not in fixedlistex
            nam='PJ%.1f%s:D%.3f' % (J_set[jset],parity, E_poles[jset,n])
            newname[fixednames[ip]] = nam 

# Copy parameters back into GNDS 
        jset = 0
        for Jpi in RMatrix.spinGroups:   # jset values need to be in the same order as before
            parity = '+' if pi_set[jset] > 0 else '-'
#           if True: print('J,pi =',J_set[jset],parity)
            R = Jpi.resonanceParameters.table
            rows = R.nRows
            cols = R.nColumns - 1  # without energy col
            if ReichMoore: cols -= 1
            for pole in range(rows):
#               print('Update',pole,'pole',R.data[pole][0],'to',E_poles[jset,pole])
                R.data[pole][0] = E_poles[jset,pole]
                c_start = 1
                if ReichMoore:
                    if IFG:
                        R.data[pole][1] = math.sqrt(D_poles[jset,pole]/2.)
                    else:
                        R.data[pole][1] = D_poles[jset,pole]
                    c_start = 2
                for c in range(cols):
                    R.data[pole][c+c_start] = g_poles[jset,pole,c]
#                 if verbose: print('\nJ,pi =',J_set[jset],parity,"revised R-matrix table:", "\n".join(R.toXMLList()))
            jset += 1
                
    print('\nR-matrix parameters:\n')
    
    print("Formal and 'Observed' properties by first-order (Thomas) corrections:")
    L_poles    = numpy.zeros([n_jsets,n_poles,n_chans,2], dtype=REAL)
    dLdE_poles = numpy.zeros([n_jsets,n_poles,n_chans,2], dtype=REAL)
    Pole_Shifts(L_poles,dLdE_poles, E_poles,has_widths, seg_val,1./cm2lab[ipair],QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
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
        
    if not Search:
        fmt = '%4i %4i   S: %10.5f %14.2f   %s'
        print('   P  Loc   Start:    V           grad    Parameter')
        for p in range(border[4]):   
            sp = searchpars0[p]; sg = grad0[p]
            if p >= border[2]:  # norms and damping
                sg /= 2.*sp
                sp = sp**2
            print(fmt % (p,searchloc[p,0],sp,sg,searchnames[p]) )

        fmt = '%4i %4i   S: %10.5f                  %s'
        print('\n   P  Loc   Fixed:    V                   Parameter')
        for ifixed in range(frontier[4]):
            sp = fixedpars[ifixed]
            print(fmt % (p,fixedloc[ifixed,0],sp, fixednames[ifixed]) )

#         ww = numpy.sum(searchpars0[border[1]:border[2]]**4) * widthWeight
        print('\n*** chisq/pt =',chisq_n/n_data, ' so chisq/dof=',chisq_n/n_dof,' with ww',ww/n_dof,' so data chisq/dof',(chisq_n-ww)/n_dof)
        covarianceSuite = None
        
    else:
        fmt = '%4i %4i   S: %10.5f %14.2f  F:  %10.5f %10.3f  %10.5f   %8.1f %%   %15s     %s'
        print('   P  Loc   Start:    V           grad    Final:     V      grad        1sig   Percent error     Parameter        new name')
        if frontier[4]>0: print('Varying:')
        for p in range(n_pars):   
            sig = inverse_hessian[p,p]**0.5
            sp0 = searchpars0[p]; sg0 = grad0[p]
            if p >= border[2]:  # norms and damping
                sg0 /= 2.*sp0
                sp0 = sp0**2
            sp1 = searchpars_n[p]; sg1 = grad1[p]
            if p >= border[2]:
                sg1 /= 2.*sp1
                sp1 = sp1**2
            newnam = newname.get(searchnames[p],'')
            if newnam == searchnames[p]: newnam = ''  # don't repeat unchanged old name
            print(fmt % (p,searchloc[p,0],sp0,sg0,sp1,sg1,sig, 100* sig/(searchpars_n[p]+1e-10),searchnames[p],newnam ) )
        fmt2 = '%4i %4i   S: %10.5f   %s     %s'
        if frontier[4]>0: print('Fixed:')
        for p in range(frontier[4]):
            print(fmt2 % (p,fixedloc[p,0],fixedpars[p],fixednames[p],newname.get(fixednames[p],'')) )
            
        print('New names for fixed parameters: ',' '.join([newname.get(fixednames[p],'') for p in range(frontier[3])]))

        print('\n*** chisq/pt = %12.5f, with chisq/dof= %12.5f for dof=%i from %11.3e' % (chisq_n/n_data,chisqpdof,n_dof,chisq_n*n_data))
                

# write Ryaml file with complete parameter+norms covariance data
    
        outFile = "%s-fit.Ryaml" % base
        write_Ryaml(gnd,outFile,inverse_hessian,border,frontier,GNDS_var,searchloc,norm_info,norm_refs,fixedloc, 
                    searchnames,searchpars_n,fixedpars,fixednames,verbose,debug)
    
# Copy covariance matrix back into GNDS 
        covarianceSuite = write_gnds_covariances(gnd,searchpars_n,inverse_hessian,GNDS_loc,POLE_details,searchnames,border,  base,verbose,debug)
                            
        trace = open('%s/bfgs_min.trace'% (base),'r')
        tracel = open('%s/bfgs_min.tracel'% (base),'w')
        traced = open('%s/bfgs_min.traced'% (base),'w')
        traces = trace.readlines( )
        trace.close( )
        lowest_chisq = 1e8
        for i,cs in enumerate(traces):
            css = cs.split()
            chis = float(css[0])
            lowest_chisq = min(lowest_chisq, chis)
            print(i+1,lowest_chisq,' '.join(css[1:3]),chis, file=tracel)
            print(i+1,css[2],css[1],chis,lowest_chisq, file=traced)
        tracel.close()
        traced.close()
    
        snap = open('%s/bfgs_min.snap'% (base),'r')
        snapl = open('%s/bfgs_min.snapl'% (base),'w')
        snaps = snap.readlines( )
        snap.close( )
        included = numpy.zeros(n_pars, dtype=INT)
        lowest_chisq = 1e6
        for vals in snaps:
#         for i,vals in enumerate(snaps):
            val_list = vals.replace('[',' ').replace(']',' ').split()
            chisqr = float(val_list[0])
            if chisqr < lowest_chisq:
                for iv,v in enumerate(val_list[3:]):
                    if abs(float(v)) > large: included[iv] = True
#                 print('Chisq at',i,'down to',lowest_chisq/n_data)
            lowest_chisq = min(lowest_chisq,chisqr)
        n_largest = numpy.count_nonzero(included)
        p_largest = []
        for i in range(n_pars):
           if included[i]:  p_largest.append(i)
        print('List the',n_largest,' parameters above',large,':\n',p_largest)

        lowest_chisq = 1e6
        for i,vals in enumerate(snaps):
            val_list = vals.replace('[',' ').replace(']',' ').split()
            chisqr = float(val_list[0])
            if chisqr < lowest_chisq:
                out = ''
                for p in p_largest:
                    out += ' ' + val_list[p+1]
                print(i,out, file=snapl)
            lowest_chisq = min(lowest_chisq,chisqr)
        snapl.close()                
    
#         print('\n*** chisq/pt = %12.5f including ww %12.5f and chisq/dof= %12.5f  for dof = %s\n' % (chisq_n/n_dof,ww/n_dof,(chisq_n - ww)/n_dof,n_dof) )
      
        n_normsFitted = border[3]-border[2]
        docLines = [' ','Fitted by Rflow','   '+inFile,str(now()),pwd.getpwuid(os.getuid())[4],' ',' ']
        docLines += [' Initial chisq/pt: %12.5f' % (chisq0_n/n_data)]
        docLines += [' Final   chisq/pt: %12.5f including ww/dof %12.5f and Chisq/DOF = %12.5f  for dof = %s\n' % (chisq_n/n_dof,ww/n_dof,(chisq_n - ww)/n_dof,n_dof) ,' ']
        docLines += [' Fitted norm %12.5f for %s' % (searchpars_n[n+border[2]],searchnames[n+border[2]] ) for n in range(n_normsFitted)] 
        docLines += [' '] 
    
        code = 'Fit quality'
        codeLabels = [item.keyValue for item in RMatrix.documentation.computerCodes]
        for i in range(2,100):
            codeLabel = '%s %s' % (code,i)
            if codeLabel not in codeLabels: break
        print('\nNew computerCode is "%s" after' % codeLabel,codeLabels,'\n')

        computerCode = computerCodeModule.ComputerCode( label = codeLabel, name = 'Rflow', version = '') #, evaluationDate = now() )
        computerCode.note.body = '\n'.join( docLines )
        RMatrix.documentation.computerCodes.add( computerCode )

    print('\n*** chisq/pt = %12.5f including ww/dof = %12.5f and chisq/dof= %12.5f  for dof = %s\n' % (chisq_n/n_dof,ww/n_dof,(chisq_n - ww)/n_dof,n_dof) )
    
    ch_info = [pname,tname, za,zb, npairs,cm2lab,QI,ipair]
    if Cross_Sections:
        return(chisq_n,ww,A_tF_n,norm_val,n_pars,n_dof,XS_totals,ch_info,covarianceSuite)
    else:
        return(chisq_n,ww,None,  norm_val,n_pars,n_dof,None,     ch_info,covarianceSuite)
