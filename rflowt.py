#!/usr/bin/env python3

import times
tim = times.times()

import os,math,numpy,cmath,pwd,sys,time,json

from CoulCF import cf1,cf2,csigma,Pole_Shifts
from evaluate_tf import evaluate_tf
from wrapup import plotOut,saveNorms2gnds
from printExcitationFunctions import *

from pqu import PQU as PQUModule
from numericalFunctions import angularMomentumCoupling
from xData.series1d  import Legendre

from fudge.gnds import reactionSuite as reactionSuiteModule
from fudge.gnds import styles        as stylesModule
from xData.Documentation import documentation as documentationModule
from xData.Documentation import computerCode  as computerCodeModule
from functools import singledispatch

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(numpy.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return numpy.float64(val)

REAL = numpy.double
CMPLX = numpy.complex128
INT = numpy.int32
realSize = 8  # bytes

print("First imports done rflow: ",tim.toString( ))


# TO DO:
#   Use nch to set search parameters even if widths=0 (i.e. decide on filler vs real level info)
#   Reich-Moore widths to imag part of E_pole like reconstructxs_TF.py
#   Multiple GPU strategies
#   Estimate initial Hessian by 1+delta parameter shift. Try various delta to make BFGS search smoother
#   Options to set parameter and search again.

# Search options:
#   Fix or search on Reich-Moore widths
#   FIXING NORMS ; FREE NORMS. Many errors in pipeline
#   Command input, e.g. as with Sfresco?

# Maybe:
#   Fit specific Legendre orders

# Doing:

##############################################  Rflow

hbc =   197.3269788e0             # hcross * c (MeV.fm)
finec = 137.035999139e0           # 1/alpha (fine-structure constant)
amu   = 931.4940954e0             # 1 amu/c^2 in MeV

coulcn = hbc/finec                # e^2
fmscal = 2e0 * amu / hbc**2
etacns = coulcn * math.sqrt(fmscal) * 0.5e0
pi = 3.1415926536
rsqr4pi = 1.0/(4*pi)**0.5

def Rflow(gnd,partitions,base,projectile4LabEnergies,data_val,data_p,n_angles,n_angle_integrals,
        Ein_list, fixedlist, emind,emaxd,pmin,pmax,
        norm_val,norm_info,norm_refs,effect_norm, LMatrix,batches,
        Search,Iterations,restarts,Distant,Background,ReichMoore, 
        verbose,debug,inFile,fitStyle,tag,large):
        
#     global L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,n_totals,brune,S_poles,dSdE_poles,EO_poles, searchloc,border, data_val, norm_info,effect_norm, Pleg, AA, chargedElastic, Rutherford, InterferenceAmpl, Gfacc
    global L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,n_totals,brune,S_poles,dSdE_poles,EO_poles, searchloc,border, Pleg, AA, chargedElastic, Rutherford, InterferenceAmpl, Gfacc

    PoPs = gnd.PoPs
    projectile = gnd.PoPs[gnd.projectile]
    target     = gnd.PoPs[gnd.target]
    elasticChannel = '%s + %s' % (gnd.projectile,gnd.target)
    if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
    if hasattr(target, 'nucleus'):     target = target.nucleus
    pZ = projectile.charge[0].value; tZ =  target.charge[0].value
    chargedElastic =  pZ*tZ != 0
    identicalParticles = gnd.projectile == gnd.target
    rStyle = fitStyle.label
#     if debug: print("Charged-particle elastic:",chargedElastic,",  identical:",identicalParticles,' rStyle:',rStyle)
    
    rrr = gnd.resonances.resolved
    Rm_Radius = gnd.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')
    bndx = RMatrix.boundaryCondition
    IFG = RMatrix.reducedWidthAmplitudes
    Overrides = False
    brune = bndx=='Brune'
    if brune: LMatrix = True
#     if brune and not LMatrix:
#         print('Brune basis requires Level-matrix method')
#         LMatrix = True
 
    n_data = data_val.shape[0]
    n_totals = n_data - n_angles - n_angle_integrals
    n_angle_integrals0 = n_angles                # so [n_angle_integrals0,n_totals0] for angle-integrals
    n_totals0 = n_angles + n_angle_integrals     # so [n_totals0:n_data]             for totals

    n_jsets = len(RMatrix.spinGroups)
    n_poles = 0
    n_chans = 0
    
    np = len(RMatrix.resonanceReactions)
    ReichMoore = False
    if RMatrix.resonanceReactions[0].eliminated: 
        print('Exclude Reich-Moore channel')
        ReichMoore = True
        np -= 1   # exclude Reich-Moore channel here
        
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
    for partition in RMatrix.resonanceReactions:
        kp = partition.label
        if partition.eliminated:  
            partitions[kp] = None
            continue
        channels[pair] = kp
        reaction = partition.reactionLink.link
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
    npairs  = pair
    if not IFG:
        print("Not yet coded for IFG =",IFG)
        sys.exit()
    
#  FIRST: for array sizes:
#
# Find energies in the lab frame of partition 'ipair' as needed for the R-matrix pole energies, not data lab frame:
#
    print('Transform main energy vector from',pname[dlabpair],'to',pname[ipair],' projectile lab frames')
    data_val[:,0]  = (data_val[:,0]/cm2lab[dlabpair] - QI[dlabpair] + QI[ipair] ) * cm2lab[ipair]
    E_scat  = data_val[:,0]
    print('Transformed E:',E_scat[:4])
    if debug: print('Energy grid (lab in partition',ipair,'):\n',E_scat)
    Elarge = 0.0
    nExcluded = 0
    if emind is not None: emin = emind
    if emaxd is not None: emax = emaxd
    for i in range(n_data):
        if not max(emin,Elarge) <= E_scat[i] <= emax:
            # print('Datum at energy %10.4f MeV outside evaluation range [%.4f,%.4f]' % (E_scat[i],emin,emax))
            Elarge = E_scat[i]
            nExcluded += 1
    if nExcluded > 0: print('\n %5i points excluded as outside range [%s, %s]' % (nExcluded,emin,emax))
    
    mu_val = data_val[:,1]

    Lmax = 0
    for Jpi in RMatrix.spinGroups:
        R = Jpi.resonanceParameters.table
        n_poles = max(n_poles,R.nRows)
        n = R.nColumns-1
        if ReichMoore: n -= 1
        n_chans = max(n_chans,n)
        for ch in Jpi.channels:
            Lmax = max(Lmax,ch.L)
    print('Need %i energies in %i Jpi sets with %i poles max, and %i channels max. Lmax=%i' % (n_data,n_jsets,n_poles,n_chans,Lmax))

    nch = numpy.zeros(n_jsets, dtype=INT)
    npl = numpy.zeros(n_jsets, dtype=INT)
    E_poles = numpy.zeros([n_jsets,n_poles], dtype=REAL)
    E_poles_fixed = numpy.zeros([n_jsets,n_poles], dtype=REAL)    # fixed in search
    has_widths = numpy.zeros([n_jsets,n_poles], dtype=INT)
    g_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=REAL)
    g_poles_fixed = numpy.zeros([n_jsets,n_poles,n_chans], dtype=REAL) # fixed in search
    J_set = numpy.zeros(n_jsets, dtype=REAL)
    pi_set = numpy.zeros(n_jsets, dtype=INT)
    L_val  =  numpy.zeros([n_jsets,n_chans], dtype=INT)
    S_val  =  numpy.zeros([n_jsets,n_chans], dtype=REAL)
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
    CS_diag = numpy.zeros([n_data,n_jsets,n_chans], dtype=CMPLX)
    Spins = [set() for pair in range(npairs)]

    
## DATA

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
    tot_channels = 0; tot_poles = 0
    for Jpi in RMatrix.spinGroups:
        J_set[jset] = Jpi.spin
        pi_set[jset] = Jpi.parity
        parity = '+' if pi_set[jset] > 0 else '-'
        R = Jpi.resonanceParameters.table
        cols = R.nColumns - 1  # ignore energy col
        rows = R.nRows
        nch[jset] = cols
        npl[jset] = rows
        if True: print('J,pi =%5.1f %s, channels %3i, poles %3i' % (J_set[jset],parity,cols,rows) )
        tot_channels += cols
        tot_poles    += rows
        seg_col[jset] = cols
        E_poles[jset,:rows] = numpy.asarray( R.getColumn('energy','MeV') , dtype=REAL)   # lab MeV
        widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']

#         if verbose:  print("\n".join(R.toXMLList()))       
        n = None
        c = 0
        All_spins = set()
        for ch in Jpi.channels:
            rr = ch.resonanceReaction
            pair = partitions.get(rr,None)
            if pair is None: continue
            m = ch.columnIndex - 1
            g_poles[jset,:rows,c] = numpy.asarray(widths[m][:],  dtype=REAL) 
            L_val[jset,c] = ch.L
            S = float(ch.channelSpin)
            S_val[jset,c] = S
            has_widths[jset,:rows] = 1
            
            seg_val[jset,c] = pair
            p_mask[pair,jset,c] = 1.0
            Spins[pair].add(S)
            All_spins.add(S)

        # Find S and P:
            for ie in range(n_data):

                if bndx == 'L' or bndx == '-L':
                    B = -ch.L
                elif bndx == 'Brune':
                    pass
                elif bndx == 'S' or bndx is None:
                    bndx = None
                elif bndx is not None:              # btype='B'
                    B = float(bndx)
                if ch.boundaryConditionOverride is not None:
                    B = float(ch.boundaryConditionOverride)

                DL = CF2_val[ie,pair,ch.L]
                S = DL.real
                P = DL.imag
                F = CF1_val[ie,pair,ch.L]
                Psr = math.sqrt(abs(P))
                phi = - math.atan2(P, F - S)
                Omega = cmath.exp(complex(0,phi))
                if bndx is None:
                    L_diag[ie,jset,c]       = complex(0.,P)
                elif bndx == 'Brune':
                    L_diag[ie,jset,c]       = DL
                else:
                    L_diag[ie,jset,c]       = DL - B

                POm_diag[ie,jset,c]      = Psr * Omega
                Om2_mat[ie,jset,c,c]     = Omega**2
                CS_diag[ie,jset,c]       = Csig_exp[ie,pair,ch.L]
            c += 1
        if debug:
            print('J set %i: E_poles \n' % jset,E_poles[jset,:])
            print('g_poles \n',g_poles[jset,:,:])
        jset += 1   

    print(' Total channels',tot_channels,' and total poles',tot_poles,'\n')

    if brune:  # S_poles: Shift functions at pole positions for Brune basis   
        S_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=REAL)
        dSdE_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=REAL)
#         EO_poles =  numpy.zeros([n_jsets,n_poles], dtype=REAL) 
        EO_poles = E_poles.copy()
        Pole_Shifts(S_poles,dSdE_poles, EO_poles,has_widths, seg_val,1./cm2lab[ipair],QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
    else:
        S_poles = None
        dSdE_poles = None
        EO_poles = None

        
    if debug:
        print('All spins:',All_spins)
        print('All channel spins',Spins)
#     print('E_poles \n',E_poles[:,:])
#     print('g_poles \n',g_poles[:,:,:])
#     print('norm_val \n',norm_val[:])

    n_norms = norm_val.shape[0]
    n_Epoles_z = numpy.count_nonzero(E_poles != 0 ) 
    n_Epoles = numpy.count_nonzero( (E_poles != 0) ) #& (abs(E_poles) < Distant) ) 
    n_gpoles = numpy.count_nonzero(g_poles != 0 ) 
    z_gpoles = numpy.count_nonzero(g_poles == 0 ) 
    n_pars  = n_Epoles+n_gpoles+n_norms
    n_Efixed = n_Epoles_z - n_Epoles
    print('Variable E,w (non-zero, non-Distant) norms:',n_Epoles,n_gpoles,n_norms,' =',n_pars,'  with',n_Efixed,'E fixed:') 
    print('Variable fixed list:',fixedlist)
    print('# zero widths  =',z_gpoles)
    searchnames = []
    searchparms = numpy.zeros(n_pars, dtype=REAL)
    searchloc  = numpy.zeros([n_pars,1], dtype=INT)   
    fixednames = []
    fixedpars = numpy.zeros(n_pars+z_gpoles, dtype=REAL)
    fixedloc  = numpy.zeros([n_pars+z_gpoles,1], dtype=INT)   

    
    ip = 0
    ifixed = 0
    border = numpy.zeros(3, dtype=INT)     # variable parameters
    frontier = numpy.zeros(3, dtype=INT)   # fixed parameters
    patterns = [ re.compile(fix_regex) for fix_regex in fixedlist] 
    fixedlistex = set()
    
    for jset in range(n_jsets):
        parity = '+' if pi_set[jset] > 0 else '-'
        for n in range(n_poles):
            i = jset*n_poles+n
            E = E_poles[jset,n]
            if E == 0: continue   # invalid energy: filler
            nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E)
            varying = abs(E) < Distant  # and n < npl[jset]
            if pmin is not None and pmax is not None and pmin > pmax: 
                varying = E > pmin or E < pmax
            else:
                if pmin is not None: varying = varying and E > pmin
                if pmax is not None: varying = varying and E < pmax
            for pattern in patterns:
                 varying = varying and not pattern.match(nam) 
#             print('Pole',jset,n,'named',nam,'at',E, 'vary:',varying)
            if varying: 
                searchparms[ip] = E
                searchloc[ip,0] = i
                searchnames += [nam]
                ip += 1
            else:
                fixedlistex.add(nam)
                E_poles_fixed[jset,n] = E_poles[jset,n]
                if Search:
                    print('    Fixed %5.1f%1s pole %2i at E = %7.3f MeV' % (J_set[jset],parity,n,E) )
                if nam not in fixedlistex and Background:
                    nam='BG:%.1f%s' % (J_set[jset],parity)
                # print('E[',jset,n,'] is fixed',ifixed,'at',E_poles[jset,n])
                fixedpars[ifixed] = E_poles[jset,n]
                fixedloc[ifixed,0] = i
                fixednames += [nam]
                ifixed += 1

    border[0] = ip
    frontier[0] = ifixed
    for jset in range(n_jsets):
        parity = '+' if pi_set[jset] > 0 else '-'
        for n in range(n_poles):
            E = E_poles[jset,n]
            Ecm = E/cm2lab[ipair] + QI[ipair]
            for c in range(n_chans):
                if L_val[jset,c] < 0: continue   # invalid partial wave: blank filler
                if E == 0: continue   # invalid energy: filler
                i = (jset*n_poles+n)*n_chans+c
                if abs(E) < Distant or not Background:
                    nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E)
                else:
                    nam='BG:%.1f%s' % (J_set[jset],parity)
                wnam = 'w'+str(c)+','+ (nam[1:] if nam[0]=='P' else nam)
                varying = abs(g_poles[jset,n,c])>1e-20 
#                 varying = c <= nch[jset] and n <= n_poles[jset] 
                if pmin is not None and pmax is not None and pmin > pmax:   # -p,-P fix both energies and widths
                    varying = Ecm > pmin or Ecm < pmax
                else:
                    if pmin is not None: varying = varying and Ecm > pmin
                    if pmax is not None: varying = varying and Ecm < pmax

                for pattern in patterns:
                    matching = pattern.match(wnam)
                    varying = varying and not pattern.match(wnam)   
#                     print('     varying=',varying,'after',wnam,'matches',pattern.match(wnam),matching,True if matching else False,pattern)              
                
#                 print('Width',jset,n,c,'named',wnam,'from',nam,E, 'vary:',varying,'\n')
                if varying:
                    searchparms[ip] = g_poles[jset,n,c]
                    searchloc[ip,0] = i
                    searchnames += [wnam]
                    ip += 1
                else:   # fixed
                    fixedlistex.add(wnam)
                    # print('g[',jset,n,c,'] is fixed',ifixed,'at',g_poles[jset,n,c])
                    fixedpars[ifixed] = g_poles[jset,n,c]
                    g_poles_fixed[jset,n,c] = g_poles[jset,n,c]
                    fixedloc[ifixed,0] = i
                    fixednames += [wnam]
                    ifixed += 1
                    
    border[1] = ip
    border[2] = border[1] + n_norms
    frontier[1] = ifixed
    frontier[2] = frontier[1] + 0  # no fixed norms yet.
    print('Variable borders:',border,'and Fixed frontiers:',frontier)
    searchparms[border[1]:border[2]] = numpy.sqrt( norm_val )   # search on p = n**0.5 so n=p^2 is always positive

    for n in range(n_norms):
        searchnames += [norm_refs[n][0]]
    n_pars = border[2]
    ndof = n_data - n_pars
        
#     print('\n Search variables:',' '.join(searchnames)) 
    print('Variable fixed list expanded:',fixedlistex)
    print('\n',len(fixednames),' fixed parameters:',' '.join(fixednames)) 
    
    if brune and False:
        for jset in range(n_jsets):
            for n in range(npl[jset]):
                print('j/n=',jset,n,' E_pole: %10.6f' % EO_poles[jset,n])
                for c in range(nch[jset]):
                     print("      S, S' %10.6f, %10.6f" % (S_poles[jset,n,c],dSdE_poles[jset,n,c]))
                                 
    print('Searching on pole energies:',searchparms[:border[0]])
              
## ANGULAR-MOMENTUM ARRAYS:

    gfac = numpy.zeros([n_data,n_jsets,n_chans], dtype=REAL)
    for jset in range(n_jsets):
        for c_in in range(n_chans):   # incoming partial wave
            pair = seg_val[jset,c_in]      # incoming partition
            if pair>=0:
                denom = (2.*jp[pair]+1.) * (2.*jt[pair]+1)
                for ie in range(n_data):
                    gfac[ie,jset,c_in] = pi * (2*J_set[jset]+1) * rksq_val[ie,pair] / denom * 10.  # mb

       
    Gfacc = numpy.zeros(n_angles, dtype=REAL)    
    NL = 2*Lmax + 1
    Pleg = numpy.zeros([n_angles,NL], dtype=REAL)
    ExptAint = numpy.zeros([n_angle_integrals,npairs, npairs], dtype=REAL)
    ExptTot = numpy.zeros([n_totals,npairs], dtype=REAL)

    for ie in range(n_angles):
        pin = data_p[ie,0]
        jproj = jp[pin]
        jtarg = jt[pin]
        denom = (2.*jproj+1.) * (2.*jtarg+1)
        Gfacc[ie]    = pi * rksq_val[ie,pin] / denom  * 10.   # mb.  Likd gfac, but no (2J+1) factor
        mu = mu_val[ie]
        if abs(mu)>1.: 
            print('Data pt ',ie,data_p[ie,:],'has bad mu:',mu_val[ie])
            sys.exit()
        for L in range(NL):
            Pleg[ie,L] = Legendre(L, mu)
                        
    for ie in range(n_angle_integrals):
        pin = data_p[n_angle_integrals0+ie,0]
        pout= data_p[n_angle_integrals0+ie,1]
        ExptAint[ie,pout,pin] = 1.
        
    for ie in range(n_totals):
        pin = data_p[n_totals0+ie,0]
        ExptTot[ie,pin] = 1.
        
    if chargedElastic:
        Rutherford = numpy.zeros([n_angles], dtype=REAL)
        InterferenceAmpl = numpy.zeros([n_angles, n_jsets, n_chans], dtype=CMPLX)
        
        for ie in range(n_angles):
            pin = data_p[ie,0]
            pout= data_p[ie,1]
            if pin==pout:
                mu = mu_val[ie]
                shthsq = (1-mu) * 0.5
                jproj = jp[pin]
                jtarg = jt[pin]
                denom = (2.*jproj+1.) * (2.*jtarg+1)
                eta = eta_val[ie,pin].real
                Coulmod  = eta.real * rsqr4pi / shthsq
                CoulAmpl = Coulmod * cmath.exp(complex(0., - eta*math.log(shthsq) ))
                Rutherford[ie] = denom * Coulmod**2
            
                for jset in range(n_jsets):
                    J = J_set[jset]
                    for c in range(n_chans):
                        if seg_val[jset,c] == pin:
                            L = L_val[jset,c]
                            InterferenceAmpl[ie,jset,c] = (2*J+1) * Pleg[ie,L] * 2 * rsqr4pi * CoulAmpl.conjugate()
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
        print('AAL, AA sizes= %5.3f, %5.3f GB' % (cc*npairs**2*NL*8/1e9, cc*n_angles*8/1e9 ),'from %s*(%s*%s^2)^2 dbles' % (n_angles,n_jsets,n_chans))
        AAL = numpy.zeros([npairs,npairs, n_jsets,n_chans,n_chans, n_jsets,n_chans,n_chans ,NL], dtype=REAL)

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

                                        for c1_out in range(n_chans):
                                            if seg_val[jset1,c1_out] != pair: continue
                                            if abs(S_val[jset1,c1_out]-So) > 0.1 : continue

                                            for jset2 in range(n_jsets):
                                                J2 = J_set[jset2]
                                                for c2 in range(n_chans):
                                                    if seg_val[jset2,c2] != inpair: continue
                                                    if abs(S_val[jset2,c2]-S) > 0.1 : continue

                                                    for c2_out in range(n_chans):
                                                        if seg_val[jset2,c2_out] != pair: continue
                                                        if abs(S_val[jset2,c2_out]-So) > 0.1 : continue
        
                                                        for L in range(NL):
                                                            ZZ = ZZbar[L,iS,jset2,c2,jset1,c1] * ZZbar[L,iSo,jset2,c2_out,jset1,c1_out] 
                                                            AAL[inpair,pair, jset2,c2_out,c2, jset1,c1_out,c1,L] += phase * ZZ / pi 

    #     AA = numpy.zeros([n_angles, n_jsets,n_chans,n_chans, n_jsets,n_chans,n_chans  ], dtype=REAL)

    #     for ie in range(n_angles):
    #         pin = data_p[ie,0]
    #         pout= data_p[ie,1]
    #         for L in range(NL):
    #             AA[ie, :,:,:, :,:,:] += AAL[pin,pout, :,:,:, :,:,:, L] * Pleg[ie,L]
    else:
        AAL = None

    E_poles_fixed_v = numpy.ravel(E_poles_fixed)
    g_poles_fixed_v = numpy.ravel(g_poles_fixed)
    
    n_angle_integrals = n_data - n_totals - n_angles
    n_angle_integrals0 = n_angles                # so [n_angle_integrals0,n_totals0] for angle-integrals
    n_totals0 = n_angles + n_angle_integrals     # so [n_totals0:n_data]             for totals


    searchpars0 = searchparms
    print('Number of search parameters:',searchpars0.shape[0])

    print("To start tf: ",tim.toString( ))

################################################################    
## TENSORFLOW CALL:

    ComputerPrecisions = (REAL, CMPLX, INT)

    Channels = [ipair,nch,npl,pname,tname,za,zb,QI,cm2lab,rmass,prmax,L_val]
    CoulombFunctions_data = [L_diag, Om2_mat,POm_diag,CS_diag, Rutherford, InterferenceAmpl, Gfacc,gfac]    # batch n_data
    CoulombFunctions_poles = [S_poles,dSdE_poles,EO_poles]                                                  # batch n_jsets

    Dimensions = [n_data,npairs,n_jsets,n_poles,n_chans,n_angles,n_angle_integrals,n_totals,NL,batches]
    Logicals = [LMatrix,brune,chargedElastic, debug,verbose]

    Search_Control = [searchloc,border,E_poles_fixed_v,g_poles_fixed_v, norm_info,effect_norm,p_mask,data_p, AAL,base, Search,Iterations,restarts]

    Data_Control = [Pleg, ExptAint,ExptTot]     # batch n_angle_integrals,  n_totals  
    
    searchpars_n, chisqF_n, A_tF_n, grad1, inverse_hessian,XS_totals, chisq0_n,grad0 = evaluate_tf(ComputerPrecisions, Channels,
        CoulombFunctions_data,CoulombFunctions_poles, Dimensions,Logicals, 
        Search_Control,Data_Control, searchpars0, data_val, tim)
        
    ch_info = [pname,tname, za,zb, npairs,cm2lab,QI,ipair]
        
    print("Finished tf: ",tim.toString( ))
#  END OF TENSORFLOW CALL
################################################################
        
    if True:     
#  Write back fitted parameters into evaluation:
        E_poles = numpy.zeros([n_jsets,n_poles], dtype=REAL) 
        g_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=REAL)
        norm_val = searchpars_n[border[1]:border[2]] ** 2

        newname = {}
        for ip in range(border[0]): #### Extract parameters after previous search:
            i = searchloc[ip,0]
            jset = i//n_poles;  n = i%n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            E_poles[jset,n] = searchpars_n[ip]
            varying = abs(E_poles[jset,n]) < Distant and searchnames[ip] not in fixedlistex
            nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  searchnames[ip] not in fixedlistex and Background: nam = 'BG:%.1f%s' % (J_set[jset],parity)
#             print(ip,'j,n E',searchnames[ip],'renamed to',nam)
            newname[searchnames[ip]] = nam

        for ip in range(frontier[0]): #### Extract parameters after previous search:
            i = fixedloc[ip,0]
            jset = i//n_poles;  n = i%n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            E_poles[jset,n] = fixedpars[ip]
            varying = abs(E_poles[jset,n]) < Distant and  fixednames[ip] not in fixedlistex
            nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  fixednames[ip] not in fixedlistex and Background: nam = 'BG:%.1f%s' % (J_set[jset],parity)
#             print(ip,'j,n fixed E',fixednames[ip],'renamed to',nam)
            newname[fixednames[ip]] = nam        
                    
        for ip in range(border[0],border[1]): ##                i = (jset*n_poles+n)*n_chans+c
            i = searchloc[ip,0]
            c = i%n_chans;  n = ( (i-c)//n_chans )%n_poles; jset = ((i-c)//n_chans -n)//n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            g_poles[jset,n,c] = searchpars_n[ip]
            nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  searchnames[ip] not in fixedlistex and Background: nam = 'BG:%.1f%s' % (J_set[jset],parity)
            wnam = 'w'+str(c)+','+ (nam[1:] if nam[0]=='P' else nam)
#             print(ip,'j,n,c width',searchnames[ip],'renamed to',wnam)
            newname[searchnames[ip]] = wnam        
        
        for ip in range(frontier[0],frontier[1]): ##                i = (jset*n_poles+n)*n_chans+c
            i = fixedloc[ip,0]
            c = i%n_chans;  n = ( (i-c)//n_chans )%n_poles; jset = ((i-c)//n_chans -n)//n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            g_poles[jset,n,c] = fixedpars[ip]
            nam='PJ%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  fixednames[ip] not in fixedlistex and Background: nam = 'BG:%.1f%s' % (J_set[jset],parity)
            wnam = 'w'+str(c)+','+ (nam[1:] if nam[0]=='P' else nam)
#             print(ip,'j,n,c fixed width',fixednames[ip],'renamed to',wnam)
            newname[fixednames[ip]] = wnam        
#         print('newname:',newname)
        
# Copy back into GNDS 
        jset = 0
        for Jpi in RMatrix.spinGroups:   # jset values need to be in the same order as before
            parity = '+' if pi_set[jset] > 0 else '-'
#             if True: print('J,pi =',J_set[jset],parity)
            R = Jpi.resonanceParameters.table
            rows = R.nRows
            cols = R.nColumns - 1  # without energy col
            for pole in range(rows):
                R.data[pole][0] = E_poles[jset,pole]
                for c in range(cols):
                    R.data[pole][c+1] = g_poles[jset,pole,c]
#                 if verbose: print('\nJ,pi =',J_set[jset],parity,"revised R-matrix table:", "\n".join(R.toXMLList()))
            jset += 1
                
        print('\nR-matrix parameters:')
         
        if not Search:
            fmt = '%4i %4i   S: %10.5f %10.5f   %15s     %s'
            print('   P  Loc   Start:    V       grad    Parameter         new name')
            for p in range(n_pars):   
                newRname = newname.get(searchnames[p],'')
                if newRname == searchnames[p]: newRname = ''
                sp = searchpars0[p]; sg = grad0[p]
                if p >= border[1]:
                    sg /= 2.*sp
                    sp = sp**2
                print(fmt % (p,searchloc[p,0],sp,sg,searchnames[p],newRname) )
#             fmt2 = '%4i %4i   S: %10.5f   %s') )
            print('\n*** chisq/pt=',chisqF_n/n_data)
            
        else:
            fmt = '%4i %4i   S: %10.5f %10.5f  F:  %10.5f %10.3f  %10.5f   %8.1f %%   %15s     %s'
            print('   P  Loc   Start:    V       grad    Final:     V      grad        1sig   Percent error     Parameter        new name')
            if frontier[2]>0: print('Varying:')
            for p in range(n_pars):   
                sig = inverse_hessian[p,p]**0.5
                sp0 = searchpars0[p]; sg0 = grad0[p]
                if p >= border[1]:
                    sg0 /= 2.*sp0
                    sp0 = sp0**2
                sp1 = searchpars_n[p]; sg1 = grad1[p]
                if p >= border[1]:
                    sg1 /= 2.*sp1
                    sp1 = sp1**2
                print(fmt % (p,searchloc[p,0],sp0,sg0,sp1,sg1,sig, sig/searchpars_n[p],searchnames[p],newname.get(searchnames[p],'') ) )
            fmt2 = '%4i %4i   S: %10.5f   %s     %s'
            if frontier[2]>0: print('Fixed:')
            for p in range(frontier[2]):   
                print(fmt2 % (p,fixedloc[p,0],fixedpars[p],fixednames[p],newname.get(fixednames[p],'')) )
                
            print('New names for fixed parameters: ',' '.join([newname.get(fixednames[p],'') for p in range(frontier[2])]))

            print('\n*** chisq/pt = %12.5f, with chisq/dof= %12.5f for dof=%i from %e11.3' % (chisqF_n/n_data,chisqF_n/ndof,ndof,chisqF_n))
                    
            covariance1 = inverse_hessian
            from scipy.linalg import eigh
            eigval1,evec1 = eigh(covariance1)
            if debug:
                print("  Covariance eigenvalue     Vector")
                for kk in range(n_pars):
                    k = n_pars-kk - 1
                    print(k,"%11.3e " % eigval1[k] , numpy.array_repr(evec1[:,k],max_line_width=200,precision=3, suppress_small=True))
            else:
                print('Covariance matrix eigenvalues:\n', numpy.array_repr(eigval1[:],max_line_width=200,precision=3, suppress_small=False) ) 

            trace = open('%s/%s-bfgs_min.trace'% (base,base),'r')
            tracel = open('%s/%s-bfgs_min.tracel'% (base,base),'w')
            traces = trace.readlines( )
            trace.close( )
            lowest_chisq = 1e8
            for i,cs in enumerate(traces):
                chis = float(cs)
                lowest_chisq = min(lowest_chisq, chis)
                print(i+1,lowest_chisq,chis, file=tracel)
            tracel.close()
        
            snap = open('%s/%s-bfgs_min.snap'% (base,base),'r')
            snapl = open('%s/%s-bfgs_min.snapl'% (base,base),'w')
            snaps = snap.readlines( )
            snap.close( )
            included = numpy.zeros(n_pars, dtype=INT)
            lowest_chisq = 1e6
            for vals in snaps:
    #         for i,vals in enumerate(snaps):
                val_list = vals.replace('[',' ').replace(']',' ').split()
                chisqr = float(val_list[0])
                if chisqr < lowest_chisq:
                    for iv,v in enumerate(val_list[1:]):
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
        

          
            docLines = [' ','Fitted by Rflow','   '+inFile,time.ctime(),pwd.getpwuid(os.getuid())[4],' ',' ']
            docLines += [' Initial chisq/pt: %12.5f' % (chisq0_n/n_data)]
            docLines += [' Final   chisq/pt: %12.5f' % (chisqF_n/n_data),' /dof= %12.5f for %i' % (chisqF_n/ndof,ndof),' ']
            docLines += ['  Fitted norm %12.5f for %s' % (searchpars_n[n+border[1]],searchnames[n+border[1]] ) for n in range(n_norms)] 
            docLines += [' '] 
        
            code = 'Fit quality'
            codeLabels = [item.keyValue for item in RMatrix.documentation.computerCodes]
            for i in range(2,100):
                codeLabel = '%s %s' % (code,i)
                if codeLabel not in codeLabels: break
            print('\nNew computerCode is "%s" after' % codeLabel,codeLabels,'\n')

            computerCode = computerCodeModule.ComputerCode( label = codeLabel, name = 'Rflow', version = '', date = time.ctime() )
            computerCode.note.body = '\n'.join( docLines )
            RMatrix.documentation.computerCodes.add( computerCode )
    
        return(chisqF_n,A_tF_n,norm_val,n_pars,XS_totals,ch_info)
        
    else:
        return(chisq_n,A_tF_n,norm_val,n_pars,XS_totals,ch_info)

############################################## main

if __name__=='__main__':
    import argparse,re

    print('\nRflow-t')
    # print('\nrflow2-v1i.py\n')
    cmd = ' '.join(sys.argv[:])
    print('Command:',cmd ,'\n')

    # Process command line options
    parser = argparse.ArgumentParser(description='Compare R-matrix Cross sections with Data')
    parser.add_argument('inFile', type=str, help='The  intial gnds R-matrix set' )
    parser.add_argument('dataFile', type=str, help='Experimental data to fit' )
    parser.add_argument('normFile', type=str, help='Experimental norms for fitting' )
    parser.add_argument("-x", "--exclude", metavar="EXCL", nargs="*", help="Substrings to exclude if any string within group name")

    parser.add_argument("-F", "--Fixed", type=str, nargs="*", help="Names of variables (as regex) to keep fixed in searches")
    parser.add_argument("-n", "--normsfixed", action="store_true",  help="Fix all physical experimental norms (but not free norms)")

    parser.add_argument("-r", "--restarts", type=int, default=0, help="max restarts for search")
    parser.add_argument("-D", "--Distant", type=float, default="25",  help="Pole energy (lab) above which are all distant poles. Fixed in  searches.")
    parser.add_argument("-B", "--Background", action="store_true",  help="Include BG in name of background poles")
    parser.add_argument("-R", "--ReichMoore", action="store_true", help="Include Reich-Moore damping widths in search")
    parser.add_argument("-L", "--LMatrix", action="store_true", help="Use level matrix method if not already Brune basis")
    parser.add_argument("-g", "--groupAngles", type=int, default="1",  help="Unused. Number of energy batches for T2B transforms, aka batches")
    parser.add_argument("-a", "--anglesData", type=int, help="Max number of angular data points to use (to make smaller search). Pos: random selection. Neg: first block")
    parser.add_argument("-m", "--maxData", type=int, help="Max number of data points to use (to make smaller search). Pos: random selection. Neg: first block")
    parser.add_argument("-e", "--emin", type=float, help="Min cm energy for gnds projectile.")
    parser.add_argument("-E", "--EMAX", type=float, help="Max cm energy for gnds projectile.")
    parser.add_argument("-p", "--pmin", type=float, help="Min energy of R-matrix pole to fit, in gnds cm energy frame. Overrides --Fixed.")
    parser.add_argument("-P", "--PMAX", type=float, help="Max energy of R-matrix pole to fit. If p>P, create gap.")

    parser.add_argument("-S", "--Search", type=str, help="Search minimization method.")
    parser.add_argument("-I", "--Iterations", type=int, help="max_iterations for search")
    
    parser.add_argument(      "--Large", type=float, default="40",  help="'large' threshold for parameter progress plotts.")
    parser.add_argument("-1", "--norm1", action="store_true", help="Use norms=1 in output analysis.")
    parser.add_argument("-C", "--Cross_Sections", action="store_true", help="Output fit and data files for grace")
    parser.add_argument("-c", "--compound", action="store_true", help="Plot -M and -C energies on scale of E* of compound system")
    parser.add_argument("-M", "--Matplot", action="store_true", help="Matplotlib data in .json output files")
    parser.add_argument("-T", "--TransitionMatrix",  type=int, default=1, help="Produce cross-section transition matrix functions in *tot_a and *fch_a-to-b")
    parser.add_argument("-l", "--logs", type=str, default='', help="none, x, y or xy for plots")
    parser.add_argument(      "--datasize", type=float,  metavar="size", default="0.2", help="Font size for experiment symbols. Default=0.2")
    parser.add_argument("-t", "--tag", type=str, default='', help="Tag identifier for this run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debugging output (more than verbose)")
    parser.add_argument("-s", "--single", action="store_true", help="Single precision: float32, complex64")
    args = parser.parse_args()

    if args.single:
        REAL = numpy.float32
        CMPLX = numpy.complex64
        INT = numpy.int32
        realSize = 4  # bytes

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
            
            
    print('lab2cmi:',lab2cmi,'and lab2cmd:',lab2cmd)
    EminFound = 1e6; EmaxFound = -1e6
    if args.emin is None and args.EMAX is None:
        data_lines = f.readlines( )
        lines_excluded = 'No'
    else:
        data_lines = []
        lines_excluded= 0      
        for line in f.readlines():
            Ed = float(line.split()[0])# in lab frame of data file
            Ecm  = Ed*lab2cmd - Qvalued + Qvaluei # in cm frame of gnds projectile.
            if emin < Ecm < emax:
                data_lines.append(line)  
                EminFound = min(EminFound,Ecm)
                EmaxFound = max(EmaxFound,Ecm)
            else:
                lines_excluded += 1      
        
    n_data = len(data_lines)
    print(n_data,'data lines after lab energies defined by projectile',projectile4LabEnergies,'(',lines_excluded,'lines excluded)')
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
    if args.debug: 
        with open(args.dataFile+'-sorted','w') as fout: fout.writelines([projectile4LabEnergies] + data_lines)

    
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
    
    print('Fitted norms:',Fitted_norm)
    f = open( args.normFile )
    norm_lines = f.readlines( )
    f.close( )    
    n_norms= len(norm_lines)
    norm_val = numpy.zeros(n_norms, dtype=REAL)  # norm,step,expect,syserror
    norm_info = numpy.zeros([n_norms,2], dtype=REAL)  # norm,step,expect,syserror
    norm_refs= []    
    ni = 0
    n_cnorms = 0
    tempfixes = 0
    for l in norm_lines:
        parts = l.split()
#         print(parts)
        norm,step, name,expect,syserror,reffile = parts
        norm,step,expect,syserror = float(norm),float(step),float(expect),float(syserror)
        if args.normsfixed and  syserror > 0.: 
            tempfixes += 1
            continue
        fitted = Fitted_norm.get(name,None)
#         print('For name',name,'find',fitted)
        if fitted is not None and not args.norm1:
            print("Using previously fitted norm for %-20s: %10.5f instead of %10.5f" % (name,fitted,norm) )
            norm = fitted
        norm_val[ni] = norm
        chi_scale = 1.0/syserror if syserror > 0. else 0.0
        norm_info[ni,:] = (expect,chi_scale)
        norm_refs.append([name,reffile])
        if syserror>0: n_cnorms += 1
        ni += 1

    n_totals = n_data - n_angles - n_angle_integrals
    print('Data points:',n_data,'of which',n_angles,'are for angles,',n_angle_integrals,'are for angle-integrals, and ',n_totals,'are for totals',
        '\nData groups:',len(groups),'\nX4 groups:',len(X4groups),'\nVariable norms:',n_norms,' of which constrained:',n_cnorms,
        '\nTemporarily fixed norms:',tempfixes)
    
    effect_norm = numpy.zeros([n_norms,n_data], dtype=REAL)
    for ni in range(n_norms):
        reffile = norm_refs[ni][1]
        pattern = re.compile(reffile)
        for id in range(n_data):
            matching = pattern.match(group_list[id])
            effect_norm[ni,id] = 1.0 if matching else 0.0
#             if matching and args.debug: 
#                 print('Pattern',reffile,' ? ',group_list[id],':', matching)
    if args.debug:
        for ni in range(n_norms):
            print('norm_val[%i]' % ni,norm_val[ni],norm_info[ni,:])
#         for id in range(n_data):
#             print('VN for id',id,':',effect_norm[:,id])

    if args.Fixed is not None: 
        print('Fixed variables:',args.Fixed)
    else:
        args.Fixed = []
    print('Energy limits.   Data min,max:',args.emin,args.EMAX,'.  Poles min,max:',args.pmin,args.PMAX)

    finalStyleName = 'fitted'
    fitStyle = stylesModule.crossSectionReconstructed( finalStyleName,
            derivedFrom=gnd.styles.getEvaluatedStyle().label )

    base = args.inFile
    if args.single: base += 's'
    base += '+%s' % args.dataFile.replace('.data','')
    if len(args.Fixed) > 0:         base += '_Fix:' + ('+'.join(args.Fixed)).replace('*','@').replace('[',':').replace(']',':')
    if args.pmin       is not None: base += '-p%s' % args.pmin
    if args.PMAX       is not None: base += '-P%s' % args.PMAX
    if args.emin       is not None: base += '-e%s' % args.emin
    if args.EMAX       is not None: base += '-E%s' % args.EMAX
    if args.maxData    is not None: base += '_m%s' % args.maxData
    if args.anglesData is not None: base += '_a%s' % args.anglesData
    if args.Search     is not None: base += '+S' 
    if args.Iterations is not None: base += '_I%s' % args.Iterations
    if args.tag != '': base = base + '_'+args.tag
     
    dataDir = base 
    if args.Cross_Sections or args.Matplot or args.TransitionMatrix >= 0 : os.system('mkdir '+dataDir)
    print("Finish setup: ",tim.toString( ))
 
    chisqtot,xsc,norm_val,n_pars,XS_totals,ch_info = Rflow(
                        gnd,partitions,base,projectile4LabEnergies,data_val,data_p,n_angles,n_angle_integrals,
                        Ein_list,args.Fixed,args.emin,args.EMAX,args.pmin,args.PMAX,
                        norm_val,norm_info,norm_refs,effect_norm, args.LMatrix,args.groupAngles,
                        args.Search,args.Iterations,args.restarts,args.Distant,args.Background,args.ReichMoore,  
                        args.verbose,args.debug,args.inFile,fitStyle,'_'+args.tag,args.Large)

    print("Finish rflow call: ",tim.toString( ))
    chisqPN = chisqtot / n_data
    print('\n ChiSq/pt = %10.4f from %i points' % (chisqPN,n_data))

    XSp_tot_n,XSp_cap_n,XSp_mat_n = XS_totals
    pname,tname, za,zb, npairs,cm2lab,QI,ipair = ch_info

    EIndex = numpy.argsort(data_val[:,0])
    if args.TransitionMatrix >= 0:
        pnin,unused = printExcitationFunctions(XSp_tot_n,XSp_cap_n,XSp_mat_n, pname,tname, za,zb, npairs, base+'/'+base,n_data,data_val[:,0],data_p,EIndex,cm2lab,QI,ipair,True)
        pnin,totals = printExcitationFunctions(XSp_tot_n,XSp_cap_n,XSp_mat_n, pname,tname, za,zb, npairs, base+'/'+base,n_data,data_val[:,0],data_p,EIndex,cm2lab,QI,ipair,False)
        pnin = 'for %s' % pnin
    else:
        totals = None
        pnin = ''

    if args.Search:  
        print('Revised norms:',norm_val)
        saveNorms2gnds(gnd,docData,previousFit,computerCodeFit,n_norms,norm_val,norm_refs)

        info = '+S_' + args.tag
        newFitFile = base  + '-fit.xml'
        open( newFitFile, mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )
        print('Written new fit file:',newFitFile)
    else:
        info = '' 


    dof = n_data + n_cnorms - n_norms - n_pars
    plotOut(n_data,n_norms,dof,args, base,info,dataDir, 
        chisqtot,data_val,norm_val,norm_info,effect_norm,norm_refs, previousFit,computerCodeFit,
        groups,cluster_list,group_list,Ein_list,Aex_list,xsc,X4groups, data_p,pins, args.TransitionMatrix,
        EIndex,totals,pname,tname,args.datasize,ipair,cm2lab, emin,emax,pnin,gnd,cmd )
        
    print("Final rflow: ",tim.toString( ))
