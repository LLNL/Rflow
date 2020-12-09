#!/usr/bin/env python3


import os,math,numpy,cmath,pwd,sys,time,json
from CoulCF import cf1,cf2,csigma,Pole_Shifts
from pqu import PQU as PQUModule

from numericalFunctions import angularMomentumCoupling
from xData.series1d  import Legendre

from fudge.gnds import reactionSuite as reactionSuiteModule
from fudge.gnds import styles        as stylesModule
from xData.Documentation import documentation as documentationModule
from xData.Documentation import computerCode  as computerCodeModule

DBLE = numpy.double
CMPLX = numpy.complex128
INT = numpy.int32

# import tensorflow as tf
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

try:
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print('TF: cannot read and/or modify virtual devices')
  pass

import times
tim = times.times()

# TO DO:
#   Reich-Moore widths to imag part of E_pole like reconstructxs_TF.py
#   Angle batching of specified size (?)
#   Fit specific Legendre orders

# Search options:
#   Fix Reich-Moore widths
#   Fixing norms 

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

plcolor = {0:"black", 1:"red", 2:"green", 3: "blue", 4:"yellow", 5:"brown", 6: "grey", 7:"violet",
            8:"cyan", 9:"magenta", 10:"orange", 11:"indigo", 12:"maroon", 13:"turquoise", 14:"darkgreen"}
pldashes = {0:'solid', 1:'dashed', 2:'dashdot', 3:'dotted'}
plsymbol = {0:".", 1:"o", 2:"s", 3: "D", 4:"^", 5:"<", 6: "v", 7:">",
            8:"P", 9:"x", 10:"*", 11:"p", 12:"1", 13:"2", 14:"3"}

@tf.function
def R2T_transformsTF(g_poles,E_poles,E_scat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans):
# Now do TF:
    GL = tf.expand_dims(g_poles,2)
    GR = tf.expand_dims(g_poles,3)

    GG  = GL * GR
    GGe = tf.expand_dims(GG,0)  

    POLES = tf.reshape(E_poles, [1,n_jsets,n_poles,1,1])  # same for all energies and channel matrix
    SCAT  = tf.reshape(E_scat,  [-1,1,1,1,1])             # vary only for scattering energies

    RPARTS = GGe / (POLES - SCAT);    # print('RPARTS',RPARTS.dtype,RPARTS.get_shape())

    RMATC = tf.reduce_sum(RPARTS,2)  # sum over poles

    C_mat = tf.eye(n_chans, dtype=CMPLX) - RMATC * tf.expand_dims(L_diag,2);            # print('C_mat',C_mat.dtype,C_mat.get_shape())

    D_mat = tf.linalg.solve(C_mat,RMATC);                                               # print('D_mat',D_mat.dtype,D_mat.get_shape())

#    S_mat = Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2);
#  T=I-S
    T_mat = tf.eye(n_chans, dtype=CMPLX) - (Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2) )
    
# multiply left and right by Coulomb phases:
    T_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)
    
    return(T_mat)

@tf.function
def Ainv(g_poles,E_poles,E_scat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles):  # FOR DEBUGGING ONL,Y
# Use Level Matrix A to get T=1-S:
#     print('g_poles',g_poles.dtype,g_poles.get_shape())
    Z = tf.constant(0.0, dtype=DBLE)
    GL = tf.reshape(g_poles,[1,n_jsets,n_poles,1,n_chans]) #; print('GL',GL.dtype,GL.get_shape())
    GR = tf.reshape(g_poles,[1,n_jsets,1,n_poles,n_chans]) #; print('GR',GR.dtype,GR.get_shape())
    LDIAG = tf.reshape(L_diag,[-1,n_jsets,1,1,n_chans]) #; print('LDIAG',LDIAG.dtype,LDIAG.get_shape())
    GLG = tf.reduce_sum( GL * LDIAG * GR , 4)    # giving [ie,J,n',ncd Rf]
    
    if brune:   # add extra terms to GLG
#         print('S_poles',S_poles.dtype,S_poles.get_shape())
        SE_poles = S_poles + tf.expand_dims(tf.math.real(E_poles)-EO_poles,2) * dSdE_poles
        POLES_L = tf.reshape(E_poles, [1,n_jsets,n_poles,1,1])  # same for all energies and channel matrix
        POLES_R = tf.reshape(E_poles, [1,n_jsets,1,n_poles,1])  # same for all energies and channel matrix
        SHIFT_L = tf.reshape(SE_poles, [1,n_jsets,n_poles,1,n_chans] ) # [J,n,c] >  [1,J,n,1,c]
        SHIFT_R = tf.reshape(SE_poles, [1,n_jsets,1,n_poles,n_chans] ) # [J,n,c] >  [1,J,1,n,c]
#         print('SHIFT_L',SHIFT_L.dtype,SHIFT_L.get_shape())
#         print('POLES_L',POLES_L.dtype,POLES_L.get_shape())
#         print('SHIFT_R',SHIFT_R.dtype,SHIFT_R.get_shape())
#         print('POLES_R',POLES_R.dtype,POLES_R.get_shape())
        SCAT  = tf.reshape(E_scat,  [-1,1,1,1,1])             # vary only for scattering energies
        NUM = tf.complex(SHIFT_L,Z) * (SCAT - POLES_R) - tf.complex(SHIFT_R,Z) * (SCAT - POLES_L)  # expect [ie,J,n',n,c]
#         print('NUM',NUM.dtype,NUM.get_shape());tf.print(NUM, summarize=-1 )
        DEN = POLES_R - POLES_L
        W_offdiag = tf.math.divide_no_nan( NUM , DEN ) 
        W_diag    = tf.reshape( tf.eye(n_poles, dtype=CMPLX), [1,1,n_poles,n_poles,1]) * tf.complex(SHIFT_R,Z) 
        W = W_diag + W_offdiag
        GLG = GLG - tf.reduce_sum( GL * W * GR , 4)

    POLES = tf.reshape(E_poles, [1,n_jsets,n_poles,1])  # same for all energies and channel matrix
    SCAT  = tf.reshape(E_scat,  [-1,1,1,1])             # vary only for scattering energies
    Ainv_mat = tf.eye(n_poles, dtype=CMPLX) * (POLES - SCAT) - GLG    # print('Ainv_mat',Ainv_mat.dtype,Ainv_mat.get_shape())
#     print('GLG',GLG.dtype,GLG.get_shape())
#     tf.print(GLG, summarize=-1 )
#     print('Ainv_mat',Ainv_mat.dtype,Ainv_mat.get_shape())
#     tf.print(Ainv_mat, summarize=-1 )
    return(Ainv_mat)

@tf.function
def LM2T_transformsTF(g_poles,E_poles,E_scat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles):
# Use Level Matrix A to get T=1-S:
#     print('g_poles',g_poles.dtype,g_poles.get_shape())
    GL = tf.reshape(g_poles,[1,n_jsets,n_poles,1,n_chans]) #; print('GL',GL.dtype,GL.get_shape())
    GR = tf.reshape(g_poles,[1,n_jsets,1,n_poles,n_chans]) #; print('GR',GR.dtype,GR.get_shape())
    LDIAG = tf.reshape(L_diag,[-1,n_jsets,1,1,n_chans]) #; print('LDIAG',LDIAG.dtype,LDIAG.get_shape())
    GLG = tf.reduce_sum( GL * LDIAG * GR , 4)    # giving [ie,J,n',ncd Rf]
    Z = tf.constant(0.0, dtype=DBLE)
    if brune:   # add extra terms to GLG
        SE_poles = S_poles + tf.expand_dims(tf.math.real(E_poles)-EO_poles,2) * dSdE_poles
        POLES_L = tf.reshape(E_poles, [1,n_jsets,n_poles,1,1])  # same for all energies and channel matrix
        POLES_R = tf.reshape(E_poles, [1,n_jsets,1,n_poles,1])  # same for all energies and channel matrix
        SHIFT_L = tf.reshape(SE_poles, [1,n_jsets,n_poles,1,n_chans] ) # [J,n,c] >  [1,J,n,1,c]
        SHIFT_R = tf.reshape(SE_poles, [1,n_jsets,1,n_poles,n_chans] ) # [J,n,c] >  [1,J,1,n,c]
#         print('SHIFT_L',SHIFT_L.dtype,SHIFT_L.get_shape())
#         print('POLES_L',POLES_L.dtype,POLES_L.get_shape())
#         print('SHIFT_R',SHIFT_R.dtype,SHIFT_R.get_shape())
#         print('POLES_R',POLES_R.dtype,POLES_R.get_shape())
        SCAT  = tf.reshape(E_scat,  [-1,1,1,1,1])             # vary only for scattering energies
#         NUM = SHIFT_L * (SCAT - POLES_R) - SHIFT_R * (SCAT - POLES_L)  # expect [ie,J,n',n,c]
        NUM = tf.complex(SHIFT_L,Z) * (SCAT - POLES_R) - tf.complex(SHIFT_R,Z) * (SCAT - POLES_L)  # expect [ie,J,n',n,c]
#         print('NUM',NUM.dtype,NUM.get_shape()); tf.print(NUM, summarize=-1 )
        DEN = POLES_L - POLES_R
        W_offdiag = tf.math.divide_no_nan( NUM , DEN )  
        W_diag    = tf.reshape( tf.eye(n_poles, dtype=CMPLX), [1,1,n_poles,n_poles,1]) * tf.complex(SHIFT_R,Z) 
        W = W_diag + W_offdiag
        GLG = GLG - tf.reduce_sum( GL * W * GR , 4)

    POLES = tf.reshape(E_poles, [1,n_jsets,n_poles,1])  # same for all energies and channel matrix
    SCAT  = tf.reshape(E_scat,  [-1,1,1,1])             # vary only for scattering energies
    Ainv_mat = tf.eye(n_poles, dtype=CMPLX) * (POLES - SCAT) - GLG    # print('Ainv_mat',Ainv_mat.dtype,Ainv_mat.get_shape())
#     print('GLG',GLG.dtype,GLG.get_shape())
#     tf.print(GLG, summarize=-1 )
#     print('Ainv_mat',Ainv_mat.dtype,Ainv_mat.get_shape())
#     tf.print(Ainv_mat, summarize=-1 )
    
    A_mat = tf.linalg.inv(Ainv_mat);       
    
    D_mat = tf.matmul( g_poles, tf.matmul( A_mat, g_poles) , transpose_a=True)     # print('D_mat',D_mat.dtype,D_mat.get_shape())

#    S_mat = Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2);
#  T=I-S
    T_mat = tf.eye(n_chans, dtype=CMPLX) - (Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2) )
    
# multiply left and right by Coulomb phases:
    T_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)

    return(T_mat)
            
@tf.function
def T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs):
                    
    Tmod2 = tf.math.real(  T_mat * tf.math.conj(T_mat) )   # ie,jset,a1,a2

# sum of Jpi sets:
    G_fac = tf.reshape(gfac, [-1,n_jsets,1,n_chans])
    XS_mat = Tmod2 * G_fac                          # ie,jset,a1,a2   
    
    G_fact = tf.reshape(gfac, [-1,n_jsets,n_chans])
    TOT_mat = tf.math.real(tf.linalg.diag_part(T_mat))   #  ie,jset,a  for  1 - Re(S) = Re(1-S) = Re(T)
    XS_tot  = TOT_mat * G_fact                           #  ie,jset,a
    p_mask1_in = tf.reshape(p_mask, [-1,npairs,n_jsets,n_chans] )   # convert pair,jset,a to  ie,pair,jset,a
    XSp_tot = 2. *  tf.reduce_sum( tf.expand_dims(XS_tot,1) * p_mask1_in , [2,3])     # convert ie,pair,jset,a to ie,pair by summing over jset,a

        
    p_mask_in = tf.reshape(p_mask,[1,1,npairs,n_jsets,1,n_chans])   ;# print('p_mask_in',p_mask_in.get_shape())   # 1,1,pin,jset,1,cin
    p_mask_out =tf.reshape(p_mask,[1,npairs,1,n_jsets,n_chans,1])   ;# print('p_mask_out',p_mask_out.get_shape()) # 1,pout,1,jset,cout,1
    
    XS_ext  = tf.reshape(XS_mat, [-1,1,1,n_jsets,n_chans,n_chans] ) ;# print('XS_ext',XS_ext.get_shape())
    XS_cpio =  XS_ext * p_mask_in * p_mask_out                      ;# print('XS_cpio',XS_cpio.get_shape())
    XSp_mat  = tf.reduce_sum(XS_cpio,[-3,-2,-1] )               # sum over jset,cout,cin, leaving ie,pout,pin
                            
    XSp_cap = XSp_tot - tf.reduce_sum(XSp_mat,1)  # total - sum of xsecs(pout)

    return(XSp_mat,XSp_tot,XSp_cap) 

        
@tf.function
def T2B_transformsTF(T_mat,AA, n_jsets,n_chans,n_angles,batches):

# BB[ie,L] = sum(i,j) T[ie,i]* AA[i,L,j] T[ie,j]
#  T= T_mat[:,n_jsets,n_chans,n_chans]

    # print(' AA', AA.get_shape())
    T_left = tf.reshape(T_mat[:n_angles,:,:],  [-1,n_jsets,n_chans,n_chans, 1,1,1])  #; print(' T_left', T_left.get_shape())
    T_right= tf.reshape(T_mat[:n_angles,:,:],  [-1,1,1,1, n_jsets,n_chans,n_chans])  #; print(' T_right', T_right.get_shape())
    
    TAT = AA * tf.math.real( tf.math.conj(T_left) * T_right )
#     TAT = AA * ( tf.math.real(T_left) * tf.math.real(T_right) + tf.math.imag(T_left) * tf.math.imag(T_right) )

    Ax = tf.reduce_sum(TAT,[ 1,2,3, 4,5,6])    # exlude dim=0 (ie)
                                                            
    return(Ax)
    
@tf.function
def T2B_transformsTFbatch(T_mat,AA, n_jsets,n_chans,n_angles,batches):

# BB[ie,L] = sum(i,j) T[ie,i]* AA[i,L,j] T[ie,j]
#  T= T_mat[:,n_jsets,n_chans,n_chans]
    if n_angles < 1:
        return ([])
        
    # batches = 5
    batch_size = n_angles //  batches + 1
    if batches>1: print(batches,'batches, so make size',batch_size)
    
    AxList  = []
    for b in range(batches):
        ie_min = b*batch_size 
        ie_max = min( (b+1)*batch_size, n_angles)
        if batches>1: print('Batch',b,'is',[ie_min,ie_max],'up to',n_angles)
        
#         T_mab = T_mat[ie_min:ie_max, :,:]
        
        T_left = tf.reshape(T_mat[ie_min:ie_max,:,:],  [-1,n_jsets,n_chans,n_chans, 1,1,1]) 
        T_right= tf.reshape(T_mat[ie_min:ie_max,:,:],  [-1,1,1,1, n_jsets,n_chans,n_chans])  

        TAT = AA[ie_min:ie_max, :,:,:, :,:,:] * tf.math.real( tf.math.conj(T_left) * T_right )
        
        Axb = tf.reduce_sum(TAT,[ 1,2,3, 4,5,6])
        AxList.append(Axb)
        
        
    Ax = tf.concat(AxList, 0)
                                                            
    return(Ax)
    
                    
@tf.function
def AddCoulombsTF(A_t,  Rutherford, InterferenceAmpl, T_mat, Gfacc, n_angles):
        
    return(( A_t + Rutherford + tf.reduce_sum (tf.math.imag( InterferenceAmpl * tf.linalg.diag_part(T_mat[:n_angles,:,:]) ) , [1,2])) * Gfacc )

@tf.function
def ChiSqTF(A_t, data_val,norm_val,norm_info,effect_norm):
    
# chi from cross-sections
    one = tf.constant(1.0, dtype=DBLE)
    fac = tf.reduce_sum(tf.expand_dims(norm_val[:]-one,1) * effect_norm, 0) + one

    chi = (A_t/fac/data_val[:,4] - data_val[:,2])/data_val[:,3]
    chisq = tf.reduce_sum(chi**2)

# chi from norm_vals themselves:

    chi = (norm_val - norm_info[:,0]) * norm_info[:,1]
    chisq += tf.reduce_sum(chi**2)
    
    return (chisq)
    
    
@tf.function        
def FitStatusTF(searchpars, others):

    L_diag, Om2_mat,POm_diag,CS_diag, LMatrix,npairs,n_jsets,n_poles,n_chans,n_totals,batches,brune,S_poles,dSdE_poles,EO_poles, searchloc,border,E_poles_fixed_v,g_poles_fixed_v, data_val, norm_info,effect_norm, Pleg, AA, chargedElastic, Rutherford, InterferenceAmpl, Gfacc,ExptAint,ExptTot,G_fact,gfac,p_mask = others

#     print('S indices',searchloc.dtype,searchloc.shape,searchloc[:,0])
#     print('S updates',searchpars.dtype,searchpars.get_shape(),searchpars)
#         
    
    n_angle_integrals = n_data - n_totals - n_angles
    n_angle_integrals0 = n_angles                # so [n_angle_integrals0,n_totals0] for angle-integrals
    n_totals0 = n_angles + n_angle_integrals     # so [n_totals0:n_data]             for totals
    print('Data points:',n_data,'of which',n_angles,'are for angles,',n_angle_integrals,'are for angle-integrals, and ',n_totals,'are for totals')

                   
    E_pole_v = tf.scatter_nd (searchloc[:border[0],:] ,          searchpars[:border[0]],          [n_jsets*n_poles] )
    g_pole_v = tf.scatter_nd (searchloc[border[0]:border[1],:] , searchpars[border[0]:border[1]], [n_jsets*n_poles*n_chans] )
    norm_val = searchpars[border[1]:border[2]]
    
    E_cpoles = tf.complex(tf.reshape(E_pole_v + E_poles_fixed_v,[n_jsets,n_poles]),        tf.constant(0., dtype=DBLE)) 
    g_cpoles = tf.complex(tf.reshape(g_pole_v + g_poles_fixed_v,[n_jsets,n_poles,n_chans]),tf.constant(0., dtype=DBLE))
    E_cscat  = tf.complex(data_val[:,0],tf.constant(0., dtype=DBLE)) 
    
    if not LMatrix:
        T_mat = R2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans ) 
    else:
        T_mat = LM2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles) 
        

    Ax = T2B_transformsTF(T_mat,AA[:, :,:,:, :,:,:], n_jsets,n_chans,n_angles,batches)

    if chargedElastic:                          
        AxA = AddCoulombsTF(Ax,  Rutherford, InterferenceAmpl, T_mat, Gfacc, n_angles)
    else:
        AxA = Ax * Gfacc
  
    XSp_mat,XSp_tot,XSp_cap  = T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs)
    
    AxI = tf.reduce_sum(XSp_mat[n_angle_integrals0:n_totals0,:,:] * ExptAint, [1,2])   # sum over pout,pin
    AxT = tf.reduce_sum(XSp_tot[n_totals0:n_data,:] * ExptTot, 1)   # sum over pin

    A_t = tf.concat([AxA,AxI,AxT], 0)
    chisq = ChiSqTF(A_t, data_val,norm_val,norm_info,effect_norm)

    Grads = tf.gradients(chisq, searchpars)

    return(chisq,A_t,Grads)
    

                                    
def Rflow(gnd,partitions,base,data_val,data_p,n_angles,n_angle_integrals,Ein_list, fixedlist,norm_val,norm_info,norm_refs,effect_norm, LMatrix,batches,
        Search,Iterations,restarts,Distant,Background,ReichMoore, verbose,debug,inFile,fitStyle,tag,large):
        
#     global L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,n_totals,brune,S_poles,dSdE_poles,EO_poles, searchloc,border, data_val, norm_info,effect_norm, Pleg, AA, chargedElastic, Rutherford, InterferenceAmpl, Gfacc
    global L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,n_totals,brune,S_poles,dSdE_poles,EO_poles, searchloc,border, Pleg, AA, chargedElastic, Rutherford, InterferenceAmpl, Gfacc

    print('\nRflow')
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
    
#     print('Reconstruction emin,emax =',emin,emax,'with',n_data,'energies')
    E_scat  = data_val[:,0]
    if debug: print('Energy grid (lab):',E_scat)
    Elarge = 0.0
    nExcluded = 0
    for i in range(n_data):
        if not max(emin,Elarge) <= E_scat[i] <= emax:
            # print('Datum at energy %10.4f MeV outside evaluation range [%.4f,%.4f]' % (E_scat[i],emin,emax))
            Elarge = E_scat[i]
            nExcluded += 1
    if nExcluded > 0: print('\n %5i points excluded as outside range [%s, %s]' % (nExcluded,emin,emax))

    mu_val = data_val[:,1]

    n_jsets = len(RMatrix.spinGroups)
    n_poles = 0
    n_chans = 0
    
    np = len(RMatrix.resonanceReactions)
    ReichMoore = False
    if RMatrix.resonanceReactions[0].eliminated: 
        print('Exclude Reich-Moore channel')
        ReichMoore = True
        np -= 1   # exclude Reich-Moore channel here
    prmax = numpy.zeros(np)
    QI = numpy.zeros(np)
    rmass = numpy.zeros(np)
    za = numpy.zeros(np)
    zb = numpy.zeros(np)
    jp = numpy.zeros(np)
    pt = numpy.zeros(np)
    ep = numpy.zeros(np)
    jt = numpy.zeros(np)
    tt = numpy.zeros(np)
    et = numpy.zeros(np)
    
    channels = {}
    pair = 0
    ipair = None
    for partition in RMatrix.resonanceReactions:
        kp = partition.label
        if partition.eliminated:  
            partitions[kp] = None
            continue
        channels[pair] = kp
        reaction = partition.reactionLink.link
        p,t = partition.ejectile,partition.residual
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
            lab2cm = tMass / (pMass + tMass)
            w_factor = 1. #/lab2cm**0.5 if IFG else 1.0
            ipair = pair  # incoming
            
        jp[pair],pt[pair],ep[pair] = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try:
            jt[pair],tt[pair],et[pair] = target.spin[0].float('hbar'), target.parity[0].value, target.energy[0].pqu('MeV').value
        except:
            jt[pair],tt[pair],et[pair] = 0.,1,0.
        print(pair,":",kp,' Q =',QI[pair],'R =',prmax[pair])
        pair += 1
    if verbose: print("\nElastic channel is",elasticChannel,'so w factor=',w_factor,'as IFG=',IFG)
    npairs  = pair
    if not IFG:
        print("Not yet coded for IFG =",IFG)
        sys.exit()
    
#  FIRST: for array sizes:
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

    E_poles = numpy.zeros([n_jsets,n_poles], dtype=DBLE)
    E_poles_fixed = numpy.zeros([n_jsets,n_poles], dtype=DBLE)    # fixed in search
    has_widths = numpy.zeros([n_jsets,n_poles], dtype=INT)
    g_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
    g_poles_fixed = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE) # fixed in search
    J_set = numpy.zeros(n_jsets, dtype=DBLE)
    pi_set = numpy.zeros(n_jsets, dtype=INT)
    L_val  =  numpy.zeros([n_jsets,n_chans], dtype=INT)
    S_val  =  numpy.zeros([n_jsets,n_chans], dtype=DBLE)
    p_mask =  numpy.zeros([npairs,n_jsets,n_chans], dtype=DBLE)
    seg_val=  numpy.zeros([n_jsets,n_chans], dtype=INT) - 1 
    seg_col=  numpy.zeros([n_jsets], dtype=INT) 

    rksq_val  = numpy.zeros([n_data,npairs], dtype=DBLE)
    velocity  = numpy.zeros([n_data,npairs], dtype=DBLE)
    
    eta_val = numpy.zeros([n_data,npairs], dtype=DBLE)   # for E>0 only
    
    CF1_val =  numpy.zeros([n_data,np,Lmax+1], dtype=DBLE)
    CF2_val =  numpy.zeros([n_data,np,Lmax+1], dtype=CMPLX)
    csigma_v=  numpy.zeros([n_data,np,Lmax+1], dtype=DBLE)
    Csig_exp=  numpy.zeros([n_data,np,Lmax+1], dtype=CMPLX)
#     Shift         = numpy.zeros([n_data,n_jsets,n_chans], dtype=DBLE)
#     Penetrability = numpy.zeros([n_data,n_jsets,n_chans], dtype=DBLE)
    L_diag = numpy.zeros([n_data,n_jsets,n_chans], dtype=CMPLX)
    POm_diag = numpy.zeros([n_data,n_jsets,n_chans], dtype=CMPLX)
    Om2_mat = numpy.zeros([n_data,n_jsets,n_chans,n_chans], dtype=CMPLX)
    CS_diag = numpy.zeros([n_data,n_jsets,n_chans], dtype=CMPLX)
    Spins = [set() for pair in range(npairs)]

    
## DATA

#  Calculate Coulomb functions on data Energy Grid
    for pair in range(npairs):
        for ie in range(n_data):
            E = E_scat[ie]*lab2cm + QI[pair]
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
            velocity[ie,pair] = k.real/rmass[pair]  # ignoring factor of hbar
            
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
    for Jpi in RMatrix.spinGroups:
        J_set[jset] = Jpi.spin
        pi_set[jset] = Jpi.parity
        parity = '+' if pi_set[jset] > 0 else '-'
        if True: print('J,pi =',J_set[jset],parity)
        R = Jpi.resonanceParameters.table
        rows = R.nRows
        cols = R.nColumns - 1  # ignore energy col
        seg_col[jset] = cols
        E_poles[jset,:rows] = numpy.asarray( R.getColumn('energy','MeV') , dtype=DBLE)   # lab MeV
        widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']

#         if verbose:  print("\n".join(R.toXMLList()))       
        n = 0
        All_spins = set()
        for ch in Jpi.channels:
            rr = ch.resonanceReaction
            pair = partitions.get(rr,None)
            if pair is None: continue
            m = ch.columnIndex - 1
            g_poles[jset,:rows,n] = numpy.asarray(widths[m][:],  dtype=DBLE) * w_factor
            L_val[jset,n] = ch.L
            S = float(ch.channelSpin)
            S_val[jset,n] = S
            has_widths[jset,:rows] = 1
            
            seg_val[jset,n] = pair
            p_mask[pair,jset,n] = 1.0
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
                    L_diag[ie,jset,n]       = complex(0.,P)
                elif bndx == 'Brune':
                    L_diag[ie,jset,n]       = DL
                else:
                    L_diag[ie,jset,n]       = DL - B

                POm_diag[ie,jset,n]      = Psr * Omega
                Om2_mat[ie,jset,n,n]     = Omega**2
                CS_diag[ie,jset,n]       = Csig_exp[ie,pair,ch.L]
            n += 1
        if debug:
            print('J set %i: E_poles \n' % jset,E_poles[jset,:])
            print('g_poles \n',g_poles[jset,:,:])
        jset += 1   

    if brune:  # S_poles: Shift functions at pole positions for Brune basis   
        S_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
        dSdE_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
#         EO_poles =  numpy.zeros([n_jsets,n_poles]) 
        EO_poles = E_poles.copy()
        Pole_Shifts(S_poles,dSdE_poles, EO_poles,has_widths, seg_val,lab2cm,QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
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
    n_pars  = n_Epoles+n_gpoles+n_norms
    n_Efixed = n_Epoles_z - n_Epoles
    print('Variable E,g,n:',n_Epoles,n_gpoles,n_norms,' =',n_pars,'  with',n_Efixed,'E fixed:') 
    print('Variable fixed list:',fixedlist)
    # print('# Searchable parameters =',n_pars)
    searchnames = []
    searchpars = numpy.zeros(n_pars, dtype=DBLE)
    searchloc  = numpy.zeros([n_pars,1], dtype=INT)   
    fixednames = []
    fixedpars = numpy.zeros(n_pars, dtype=DBLE)
    fixedloc  = numpy.zeros([n_pars,1], dtype=INT)   

    
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
            nam='J%.1f%s:E%.3f' % (J_set[jset],parity, E)
            varying = abs(E) < Distant             
            for pattern in patterns:
                 varying = varying and not pattern.match(nam) 
#             print('Pole',jset,n,'named',nam,'at',E, 'vary:',varying)
            if varying: 
                searchpars[ip] = E
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
            for c in range(n_chans):
                if L_val[jset,c] < 0: continue   # invalid partial wave: blank filler
                if E == 0: continue   # invalid energy: filler
                i = (jset*n_poles+n)*n_chans+c
                if abs(E) < Distant or not Background:
                    nam='J%.1f%s:E%.3f' % (J_set[jset],parity, E)
                else:
                    nam='BG:%.1f%s' % (J_set[jset],parity)
                wnam = 'w'+str(c)+','+ (nam[1:] if nam[0]=='J' else nam)
                varying = abs(g_poles[jset,n,c])>1e-20 
                for pattern in patterns:
                    matching = pattern.match(wnam)
                    varying = varying and not pattern.match(wnam)   
#                     print('     varying=',varying,'after',wnam,'matches',pattern.match(wnam),matching,True if matching else False,pattern)              
                
#                 print('Width',jset,n,c,'named',wnam,'from',nam,E, 'vary:',varying,'\n')
                if varying:
                    searchpars[ip] = g_poles[jset,n,c]
                    searchloc[ip,0] = i
                    searchnames += [wnam]
                    ip += 1
                else:   # fixed
                    fixedlistex.add(wnam)
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
#     print('Norms: val',norm_val,'refs',norm_refs)
#     print(border[1],border[2],border[2]-border[1],searchpars[border[1]:border[2]], norm_val)
    searchpars[border[1]:border[2]] = norm_val   
    for n in range(n_norms):
        searchnames += [norm_refs[n][0]]
    n_pars = border[2]
        
#     print('\n Search variables:',' '.join(searchnames)) 
    print('Variable fixed list expanded:',fixedlistex)
    print('\n',len(fixednames),' fixed parameters:',' '.join(fixednames)) 
    
    if brune and False:
        for jset in range(n_jsets):
            for n in range(n_poles):
                print('j/n=',jset,n,' E_pole: %10.6f' % EO_poles[jset,n])
                for c in range(n_chans):
                     print("      S, S' %10.6f, %10.6f" % (S_poles[jset,n,c],dSdE_poles[jset,n,c]))
                                 
    print('Searching on pole energies:',searchpars[:border[0]])
## EVALUATE R-MATRIX:
              
    E_poles_fixed_v = numpy.ravel(E_poles_fixed)
    g_poles_fixed_v = numpy.ravel(g_poles_fixed)
        
    E_pole_v = tf.scatter_nd (searchloc[:border[0],:] , searchpars[:border[0]], [n_jsets*n_poles] )
    g_pole_v = tf.scatter_nd (searchloc[border[0]:border[1],:] , searchpars[border[0]:border[1]], [n_jsets*n_poles*n_chans] )
    norm_val = searchpars[border[1]:border[2]]

    E_poles = tf.reshape(E_pole_v + E_poles_fixed_v,[n_jsets,n_poles])
    g_poles = tf.reshape(g_pole_v + g_poles_fixed_v,[n_jsets,n_poles,n_chans])
    
    E_cpoles = tf.complex(E_poles,tf.constant(0., dtype=DBLE)) 
    g_cpoles = tf.complex(g_poles,tf.constant(0., dtype=DBLE))
    
    E_cscat = tf.complex(E_scat,tf.constant(0., dtype=DBLE)) 
    
    if not LMatrix:
        T_mat = R2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans ) 
    else:
        if debug:
            Ainv_mat = Ainv(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles ) 
            print('Ainv_mat:\n',Ainv_mat.numpy()[0,:,:,:] )
        T_mat = LM2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles ) 

    gfac = numpy.zeros([n_data,n_jsets,n_chans])
    for jset in range(n_jsets):
        for c_in in range(n_chans):   # incoming partial wave
            pair = seg_val[jset,c_in]      # incoming partition
            if pair>=0:
                denom = (2.*jp[pair]+1.) * (2.*jt[pair]+1)
                for ie in range(n_data):
                    gfac[ie,jset,c_in] = pi * (2*J_set[jset]+1) * rksq_val[ie,pair] / denom * 10.  # mb

################################################### DIAGNOSTIC XS:   ie only used for energy for full matrices.

    if debug:
#         for ie in range(n_data):
#             for jset in range(n_jsets):
#                 print('Energy',E_scat[ie],'  J=',J_set[jset],pi_set[jset],'\n R-matrix is size',seg_col[jset])
#                 for a in range(n_chans):
#                     print('   ',a,'row: ',',  '.join(['{:.5f}'.format(RMATC[ie,jset,a,b].numpy()) for b in range(n_chans)]) )
    
        for ie in range(n_data):
            for jset in range(n_jsets):
                print('Energy',E_scat[ie],'  J=',J_set[jset],pi_set[jset],'\n T-matrix is size',seg_col[jset])
                for a in range(n_chans):
                    print('   ',a,'row: ',',  '.join(['{:.5f}'.format(T_mat[ie,jset,a,b].numpy()) for b in range(n_chans)]) )

    if verbose:
        XSp_mat,XSp_tot,XSp_cap  = T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs)
                
        XSp_tot_n = XSp_tot.numpy()
        XSp_cap_n = XSp_cap.numpy()
        XSp_mat_n = XSp_mat.numpy()
        T_mat_n = T_mat.numpy()
        
        for pin in range(npairs):
            fname = base + '-tot_%i' % pin
            cname = base + '-cap_%i' % pin
            print('Total cross-sections for incoming',pin,'to file',fname,' and capture to',cname)
            fout = open(fname,'w')
            cout = open(cname,'w')
            for ie in range(n_data):
                E = E_scat[ie]      # lab incident energy
                E = Ein_list[ie]    # incident energy in EXFOR experiment
                x = XSp_tot_n[ie,pin] 
                print(E,x, file=fout)
                c = XSp_cap_n[ie,pin] 
                print(E,c, file=cout)
            fout.close()
            cout.close()

            for pout in range(npairs):
                fname = base + '-ch_%i-to-%i' % (pin,pout)
                print('Partition',pin,'to',pout,': angle-integrated cross-sections to file',fname)
                fout = open(fname,'w')
                
                for ie in range(n_data):
#                   y = 0
#                   for jset in range(n_jsets):
#                       for c_in in range(n_chans):   # incoming partial wave
#                           pair = seg_val[jset,c_in]      # incoming partition
#                           if seg_val[jset,c_in] !=pin: continue
#                           for c_out in range(n_chans):   # outgoing partial wave
#                               if seg_val[jset,c_out] !=pout: continue
#                               y += gfac[ie,jset,c_in] * abs(T_mat_n[ie,jset,c_out,c_in])**2
                                                        
                    x = XSp_mat_n[ie,pout,pin]
                    E = E_scat[ie]
                    E = Ein_list[ie]
                    print(E,x, file=fout)
                fout.close()
    
###################################################

       
    Gfacc = numpy.zeros(n_angles, dtype=DBLE)    
    G_fact = tf.reshape(gfac, [-1,n_jsets,n_chans])
    NL = 2*Lmax + 1
    Pleg = numpy.zeros([n_angles,NL])
    ExptAint = numpy.zeros([n_angle_integrals,npairs, npairs], dtype=DBLE)
    ExptTot = numpy.zeros([n_totals,npairs], dtype=DBLE)

    for ie in range(n_angles):
        pin = data_p[ie,0]
        jproj = jp[pin]
        jtarg = jt[pin]
        denom = (2.*jproj+1.) * (2.*jtarg+1)
        Gfacc[ie]    = pi * rksq_val[ie,pin] / denom  * 10.   # mb
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
        Rutherford = numpy.zeros([n_angles], dtype=DBLE)
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

    
    NS = len(All_spins)
    ZZbar = numpy.zeros([NL,NS,n_jsets,n_chans,n_jsets,n_chans])

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

    BB = numpy.zeros([n_data,NL])
    AAL = numpy.zeros([npairs,npairs, n_jsets,n_chans,n_chans, n_jsets,n_chans,n_chans ,NL], dtype=DBLE)

    for rr_in in RMatrix.resonanceReactions:
        if rr_in.eliminated: continue
        ipair = partitions[rr_in.label]

        for rr_out in RMatrix.resonanceReactions:
            if rr_out.eliminated: continue
            pair = partitions[rr_out.label]
                
            for S_out in Spins[pair]:
                for S_in in Spins[ipair]:
#                     print('>> S_in:',S_in)
                    for iS,S in enumerate(All_spins):
                        for iSo,So in enumerate(All_spins):
                            if abs(S-S_in)>0.1 or abs(So-S_out)>0.1: continue
                            phase = (-1)**int(So-S) / 4.0


                            for jset1 in range(n_jsets):
                                J1 = J_set[jset1]
                                for c1 in range(n_chans):
                                    if seg_val[jset1,c1] != ipair: continue
                                    if abs(S_val[jset1,c1]-S) > 0.1 : continue

                                    for c1_out in range(n_chans):
                                        if seg_val[jset1,c1_out] != pair: continue
                                        if abs(S_val[jset1,c1_out]-So) > 0.1 : continue

                                        for jset2 in range(n_jsets):
                                            J2 = J_set[jset2]
                                            for c2 in range(n_chans):
                                                if seg_val[jset2,c2] != ipair: continue
                                                if abs(S_val[jset2,c2]-S) > 0.1 : continue

                                                for c2_out in range(n_chans):
                                                    if seg_val[jset2,c2_out] != pair: continue
                                                    if abs(S_val[jset2,c2_out]-So) > 0.1 : continue
        
                                                    for L in range(NL):
                                                        ZZ = ZZbar[L,iS,jset2,c2,jset1,c1] * ZZbar[L,iSo,jset2,c2_out,jset1,c1_out] 
                                                        AAL[ipair,pair, jset2,c2_out,c2, jset1,c1_out,c1,L] += phase * ZZ / pi 

    AA = numpy.zeros([n_angles, n_jsets,n_chans,n_chans, n_jsets,n_chans,n_chans  ], dtype=DBLE)
    cc = (n_jsets*n_chans**2)**2
    print('AAL, AA sizes= %5.3f, %5.3f GB' % (cc*npairs**2*NL*8/1e9, cc*n_angles*8/1e9 ),'from %s*(%s*%s^2)^2 dbles' % (n_angles,n_jsets,n_chans))
    for ie in range(n_angles):
        pin = data_p[ie,0]
        pout= data_p[ie,1]
        for L in range(NL):
            AA[ie, :,:,:, :,:,:] += AAL[pin,pout, :,:,:, :,:,:, L] * Pleg[ie,L]

    Ax = T2B_transformsTF(T_mat,AA[:, :,:,:, :,:,:], n_jsets,n_chans,n_angles,batches)

#     Angular_XS= A_t.numpy()   # =RT
    
    if chargedElastic:                          
        AxA = AddCoulombsTF(Ax,  Rutherford, InterferenceAmpl, T_mat, Gfacc, n_angles)
    else:
        AxA *= Gfacc

    XSp_mat,XSp_tot,XSp_cap  = T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs)
    AxI = tf.reduce_sum(XSp_mat[n_angle_integrals0:n_totals0,:,:] * ExptAint, [1,2])   # sum over pout,pin
    
    AxT = tf.reduce_sum(XSp_tot[n_totals0:n_data,:] * ExptTot, 1)   # sum over pin

    A_t = tf.concat([AxA, AxI, AxT], 0)
#   print('Ax*',AxA.get_shape(),AxI.get_shape(),AxT.get_shape(),'giving',A_t.get_shape(),'to be used with',data_val.shape)

    chisq = ChiSqTF(A_t, data_val,norm_val,norm_info,effect_norm)
    print('\nFirst run:',chisq.numpy()/n_data)  
    
    if verbose:
        if n_angles>0: xsFile = open(base + '.xsa','w')
        Angular_XS = AxA.numpy()
        chisqsum = 0.0      
        for ie in range(n_angles):
            fac = 1.0
            for ni in range(n_norms):
                fac += (norm_val[ni]-1.)*effect_norm[ni,ie]
            chi = (Angular_XS[ie]/fac/data_val[ie,4]-data_val[ie,2])/data_val[ie,3] 
            chisqsum += chi**2
            theta = math.acos(mu_val[ie])*180./pi if mu_val[ie] <= 1.0 else -1.
        
            print('%8.2f %8.2f %10.3f %10.3f     %6.1f %6.2f' % (data_val[ie,0],theta,Angular_XS[ie]/data_val[ie,4],data_val[ie,2]*fac,chi,fac), file=xsFile)

# chi from norm_vals themselves:
        print('\nchisq/pt=',chisqsum/(n_data),'before constrained norms' )
        chisqnorms = 0.
        for ni in range(n_norms):
            chi = (norm_val[ni] - norm_info[ni,0]) * norm_info[ni,1]        
            chisqnorms += chi**2
     
        chisqsum += chisqnorms
        print('chisq/pt=',chisqsum/(n_data),'(including)' )

    searchpars0 = searchpars
    others = (L_diag, Om2_mat,POm_diag,CS_diag,    LMatrix,npairs,n_jsets,n_poles,n_chans,n_totals,batches,brune,S_poles,dSdE_poles,EO_poles,  searchloc,border,E_poles_fixed_v,g_poles_fixed_v,
                data_val, norm_info,effect_norm, Pleg, AA, chargedElastic, Rutherford, InterferenceAmpl, Gfacc,ExptAint,ExptTot,G_fact,gfac,p_mask)
#             L_diag, Om2_mat,POm_diag,CS_diag,    LMatrix,npairs,n_jsets,n_poles,n_chans,n_totals,batches,brune,S_poles,dSdE_poles,EO_poles,  searchloc,border,E_poles_fixed_v,g_poles_fixed_v, 
#               data_val, norm_info,effect_norm, Pleg, AA, chargedElastic, Rutherford, InterferenceAmpl, Gfacc,ExptAint,ExptTot,G_fact,gfac,p_mask = others

    n_angle_integrals = n_data - n_totals - n_angles
    n_angle_integrals0 = n_angles                # so [n_angle_integrals0,n_totals0] for angle-integrals
    n_totals0 = n_angles + n_angle_integrals     # so [n_totals0:n_data]             for totals
                    
#     chisqF,A_tF,Grads = FitStatusTF(searchpars, others) 
#     print('\n*** chisq/pt=',chisqF.numpy()/n_data)
    if Search:
#         trace = open('bfgs_minimize.trace','w')
        os.system("rm -f %s-bfgs_min%s.trace" % (base,tag) ) 
        os.system("rm -f %s-bfgs_min%s.snap" % (base,tag) )
        trace = "file://%s-bfgs_min%s.trace" % (base,tag) 
        snap = "file://%s-bfgs_min%s.snap"  % (base,tag) 
        ndof = n_data - border[2]
        
        @tf.function        
        def FitMeasureTF(searchpars):

            E_pole_v = tf.scatter_nd (searchloc[:border[0],:] ,          searchpars[:border[0]],          [n_jsets*n_poles] )
            g_pole_v = tf.scatter_nd (searchloc[border[0]:border[1],:] , searchpars[border[0]:border[1]], [n_jsets*n_poles*n_chans] )
            norm_val = searchpars[border[1]:border[2]]
    
            E_cpoles = tf.complex(tf.reshape(E_pole_v+E_poles_fixed_v,[n_jsets,n_poles]),        tf.constant(0., dtype=DBLE)) 
            g_cpoles = tf.complex(tf.reshape(g_pole_v+g_poles_fixed_v,[n_jsets,n_poles,n_chans]),tf.constant(0., dtype=DBLE))
            E_cscat  = tf.complex(data_val[:,0],tf.constant(0., dtype=DBLE)) 

            if not LMatrix:
                T_mat = R2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans ) 
            else:
                T_mat = LM2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles ) 

            Ax = T2B_transformsTF(T_mat,AA[:, :,:,:, :,:,:], n_jsets,n_chans,n_angles,batches)

            if chargedElastic:                          
                AxA = AddCoulombsTF(Ax,  Rutherford, InterferenceAmpl, T_mat, Gfacc, n_angles)
            else:
                AxA =  Ax * Gfacc
            
            XSp_mat,XSp_tot,XSp_cap  = T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs)
    
            AxI = tf.reduce_sum(XSp_mat[n_angle_integrals0:n_totals0,:,:] * ExptAint, [1,2])   # sum over pout,pin
            AxT = tf.reduce_sum(XSp_tot[n_totals0:n_data,:] * ExptTot, 1)   # sum over pin
            
            A_t = tf.concat([AxA, AxI, AxT], 0)
            chisq = ChiSqTF(A_t, data_val,norm_val,norm_info,effect_norm)
            
            tf.print(chisq,         output_stream=trace)
            tf.print(chisq, searchpars,  summarize=-1,   output_stream=snap)
            
            return(chisq, tf.gradients(chisq, searchpars)[0] )
    
        initial_objective = FitMeasureTF(searchpars) 
        chisq = initial_objective[0]
        grad0 = initial_objective[1].numpy()
        import tensorflow_probability as tfp   
        print('Initial position:',chisq.numpy()/n_data )
    
        optim_results = tfp.optimizer.bfgs_minimize (FitMeasureTF, initial_position=searchpars,
                            max_iterations=Iterations, tolerance=float(Search))
                            
        last_cppt = optim_results.objective_value.numpy()/n_data
        for restart in range(restarts):
            searchpars = optim_results.position.numpy()

            print('More pole energies:',searchpars[:border[0]])
            print('Before restart',restart,' objective chisq/pt',last_cppt)
            print('And objective FitMeasureTF =',FitMeasureTF(searchpars)[0].numpy()/n_data )
            
            if brune:
                EOO_poles = EO_poles.copy()
                SOO_poles = S_poles.copy()
                for ip in range(border[0]): #### Extract parameters after previous search:
                    i = searchloc[ip,0]
                    jset = i//n_poles;  n = i%n_poles
                    EO_poles[jset,n] = searchpars[ip]
                Pole_Shifts(S_poles,dSdE_poles, EO_poles,has_widths, seg_val,lab2cm,QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
                

                for jset in range(n_jsets):
                    for n in range(n_poles):
                        print('j/n=',jset,n,' E old,new:',EOO_poles[jset,n],EO_poles[jset,n])
                        for c in range(n_chans):
                             print('      S old,new %10.6f, %10.6f, expected %5.2f %%' % (SOO_poles[jset,n,c],S_poles[jset,n,c],
                                     100*dSdE_poles[jset,n,c]*(EO_poles[jset,n]-EOO_poles[jset,n])/ (S_poles[jset,n,c] - SOO_poles[jset,n,c])))
                    
                others = (L_diag, Om2_mat,POm_diag,CS_diag,    LMatrix,npairs,n_jsets,n_poles,n_chans,n_totals,batches,brune,S_poles,dSdE_poles,EO_poles,  searchloc,border,E_poles_fixed_v,g_poles_fixed_v, \
                            data_val, norm_info,effect_norm, Pleg, AA, chargedElastic, Rutherford, InterferenceAmpl, Gfacc,ExptAint,ExptTot,G_fact,gfac,p_mask)
                n_angle_integrals = n_data - n_totals - n_angles
                n_angle_integrals0 = n_angles                # so [n_angle_integrals0,n_totals0] for angle-integrals
                n_totals0 = n_angles + n_angle_integrals     # so [n_totals0:n_data]             for totals
                                
                T_mat = LM2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles ) 

                XSp_mat,XSp_tot,XSp_cap  = T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs)
                
                AxA = T2B_transformsTF(T_mat,AA[:, :,:,:, :,:,:], n_jsets,n_chans,n_angles,batches)
                AxA = AddCoulombsTF(AxA,  Rutherford, InterferenceAmpl, T_mat, Gfacc, n_angles)
                
                XSp_mat,XSp_tot,XSp_cap  = T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs)
    
                AxI = tf.reduce_sum(XSp_mat[n_angle_integrals0:n_totals0,:,:] * ExptAint, [1,2])   # sum over pout,pin
                AxT = tf.reduce_sum(XSp_tot[n_totals0:n_data,:] * ExptTot, 1)   # sum over pin
                    
                A_t = tf.concat([AxA, AxI, AxT], 0) 
    
                chisq = ChiSqTF(A_t, data_val,norm_val,norm_info,effect_norm)
                print('And ChiSqTF again =',chisq.numpy()/n_data )
                
            optim_results = tfp.optimizer.bfgs_minimize (FitMeasureTF, initial_position=searchpars,
                    max_iterations=Iterations, tolerance=float(Search))
            new_cppt = optim_results.objective_value.numpy()/n_data
            if new_cppt >= last_cppt: break
            last_cppt = new_cppt
                      
        print('\nResults:')
        print('Converged:',optim_results.converged.numpy(), 'Failed:',optim_results.failed.numpy())
        print('Num_iterations:',optim_results.num_iterations.numpy(), 'Num_objective_evaluations:',optim_results.num_objective_evaluations.numpy())
        print('Objective_value:',optim_results.objective_value.numpy(), 'Objective chisq/pt',optim_results.objective_value.numpy()/n_data)
        if not verbose: print('position:',optim_results.position.numpy())
        inverse_hessian = optim_results.inverse_hessian_estimate.numpy()
        if not verbose: print('inverse_hessian: shape=',inverse_hessian.shape ,'\ndiagonal:',[inverse_hessian[i,i] for i in range(n_pars)] )
        searchpars = optim_results.position.numpy()
        norm_val = searchpars[border[1]:border[2]]
        

    if True:     
        # chisqF = FitMeasureTF(searchpars) [0]
        # print('\nchisq from FitMeasureTF:',chisqF.numpy())
        
        chisqF,A_tF,Grads = FitStatusTF(searchpars, others) 
        print(  'chisq from FitStatusTF:',chisqF.numpy())
        
#  Write back fitted parameters into evaluation:
        E_poles = numpy.zeros([n_jsets,n_poles], dtype=DBLE) 
        g_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
            
        newname = {}
        for ip in range(border[0]): #### Extract parameters after previous search:
            i = searchloc[ip,0]
            jset = i//n_poles;  n = i%n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            E_poles[jset,n] = searchpars[ip]
            varying = abs(E_poles[jset,n]) < Distant and searchnames[ip] not in fixedlistex
            nam='J%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  searchnames[ip] not in fixedlistex and Background: nam = 'BG:%.1f%s' % (J_set[jset],parity)
#             print(ip,'j,n E',searchnames[ip],'renamed to',nam)
            newname[searchnames[ip]] = nam

        for ip in range(frontier[0]): #### Extract parameters after previous search:
            i = fixedloc[ip,0]
            jset = i//n_poles;  n = i%n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            E_poles[jset,n] = fixedpars[ip]
            varying = abs(E_poles[jset,n]) < Distant and  fixednames[ip] not in fixedlistex
            nam='J%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  fixednames[ip] not in fixedlistex and Background: nam = 'BG:%.1f%s' % (J_set[jset],parity)
#             print(ip,'j,n fixed E',fixednames[ip],'renamed to',nam)
            newname[fixednames[ip]] = nam        
                    
        for ip in range(border[0],border[1]): ##                i = (jset*n_poles+n)*n_chans+c
            i = searchloc[ip,0]
            c = i%n_chans;  n = ( (i-c)//n_chans )%n_poles; jset = ((i-c)//n_chans -n)//n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            g_poles[jset,n,c] = searchpars[ip]
            nam='J%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  searchnames[ip] not in fixedlistex and Background: nam = 'BG:%.1f%s' % (J_set[jset],parity)
            wnam = 'w'+str(c)+','+ (nam[1:] if nam[0]=='J' else nam)
#             print(ip,'j,n,c width',searchnames[ip],'renamed to',wnam)
            newname[searchnames[ip]] = wnam        
        
        for ip in range(frontier[0],frontier[1]): ##                i = (jset*n_poles+n)*n_chans+c
            i = fixedloc[ip,0]
            c = i%n_chans;  n = ( (i-c)//n_chans )%n_poles; jset = ((i-c)//n_chans -n)//n_poles
            parity = '+' if pi_set[jset] > 0 else '-'
            g_poles[jset,n,c] = fixedpars[ip]
            nam='J%.1f%s:E%.3f' % (J_set[jset],parity, E_poles[jset,n])
            if not varying and  fixednames[ip] not in fixedlistex and Background: nam = 'BG:%.1f%s' % (J_set[jset],parity)
            wnam = 'w'+str(c)+','+ (nam[1:] if nam[0]=='J' else nam)
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
                
#         print('gradient:\n',Grads[0].numpy())
        print('\nR-matrix parameters:')
         
        if not Search:
            grad0 = Grads[0].numpy()
            fmt = '%4i %4i   S: %10.5f %10.5f   %15s     %s'
            print('   P  Loc   Start:    V       grad    Parameter         new name')
            for p in range(n_pars):   
                newRname = newname.get(searchnames[p],'')
                if newRname == searchnames[p]: newRname = ''
                print(fmt % (p,searchloc[p,0],searchpars0[p],grad0[p],searchnames[p],newRname) )
#             fmt2 = '%4i %4i   S: %10.5f   %s') )
            print('\n*** chisq/pt=',chisqF.numpy()/n_data)
            
        else:
            grad1 = Grads[0].numpy()
            fmt = '%4i %4i   S: %10.5f %10.5f  F:  %10.5f %10.3f  %10.5f   %8.1f %%   %15s     %s'
            print('   P  Loc   Start:    V       grad    Final:     V      grad        1sig   Percent error     Parameter        new name')
            if frontier[2]>0: print('Varying:')
            for p in range(n_pars):   
                sig = inverse_hessian[p,p]**0.5
                print(fmt % (p,searchloc[p,0],searchpars0[p],grad0[p],searchpars[p],grad1[p],sig, sig/searchpars[p],searchnames[p],newname.get(searchnames[p],'') ) )
            fmt2 = '%4i %4i   S: %10.5f   %s     %s'
            if frontier[2]>0: print('Fixed:')
            for p in range(frontier[2]):   
                print(fmt2 % (p,fixedloc[p,0],fixedpars[p],fixednames[p],newname.get(fixednames[p],'')) )
                
            print('New names for fixed parameters: ',' '.join([newname.get(fixednames[p],'') for p in range(frontier[2])]))

            print('\n*** chisq/pt = %12.5f, with chisq/dof= %12.5f for dof=%i from %e11.3' % (chisqF.numpy()/n_data,chisqF.numpy()/ndof,ndof,chisqF.numpy()))
                    
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

            trace = open('%s-bfgs_min%s.trace'% (base,tag),'r')
            tracel = open('%s-bfgs_min%s.tracel'% (base,tag),'w')
            traces = trace.readlines( )
            trace.close( )
            lowest_chisq = 1e6
            for i,cs in enumerate(traces):
                chis = float(cs)/n_data
                lowest_chisq = min(lowest_chisq, chis)
                print(i+1,lowest_chisq,chis, file=tracel)
            tracel.close()
        
            snap = open('%s-bfgs_min%s.snap'% (base,tag),'r')
            snapl = open('%s-bfgs_min%s.snapl'% (base,tag),'w')
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
            docLines += [' Initial chisq/pt: %12.5f' % ( chisq.numpy()/n_data)]
            docLines += [' Final   chisq/pt: %12.5f' % (chisqF.numpy()/n_data),' /dof= %12.5f for %i' % (chisqF.numpy()/ndof,ndof),' ']
            docLines += ['  Fitted norm %12.5f for %s' % (searchpars[n+border[1]],searchnames[n+border[1]] ) for n in range(n_norms)] 
            docLines += [' '] 
        
            computerCode = computerCodeModule.ComputerCode( label = 'Fit quality', name = 'Rflow', version = '', date = time.ctime() )
            computerCode.note.body = '\n'.join( docLines )
            RMatrix.documentation.computerCodes.add( computerCode )
    
        return(chisqF.numpy(),A_tF.numpy(),norm_val,n_pars)
        
    else:
        return(chisq.numpy(),Ax.numpy(),norm_val,n_pars)

#             print('Chisq :',chisq.numpy()/n_data)

############################################## main

if __name__=='__main__':
    import argparse,re

    # print('\nrflow2-v1i.py\n')
    cmd = ' '.join(sys.argv[:])
    print('Command:',cmd ,'\n')

    # Process command line options
    parser = argparse.ArgumentParser(description='Compare R-matrix Cross sections with Data')
    parser.add_argument('inFile', type=str, help='The  intial gnds R-matrix set' )
    parser.add_argument('data', type=str, help='Experimental data to fit' )
    parser.add_argument('norm', type=str, help='Experimental norms for fitting' )
    parser.add_argument("-F", "--Fixed", type=str, nargs="*", help="Names of variables (as regex) to keep fixed in searches")
    parser.add_argument("-1", "--norm1", action="store_true", help="Use norms=1")
    parser.add_argument("-S", "--Search", type=str, help="Search minimization method.")
    parser.add_argument("-I", "--Iterations", type=int, help="max_iterations for search")
    parser.add_argument("-r", "--restarts", type=int, default=0, help="max restarts for search")
    parser.add_argument("-D", "--Distant", type=float, default="25",  help="Pole energy (lab) above which are all distant poles. Fixed in  searches.")
    parser.add_argument("-B", "--Background", action="store_true",  help="Include BG in name of background poles")
    parser.add_argument("-R", "--ReichMoore", action="store_true", help="Include Reich-Moore damping widths in search")
    parser.add_argument("-L", "--LMatrix", action="store_true", help="Use level matrix method if not already Brune basis")
    parser.add_argument("-A", "--AngleBunching", type=int, default="1",  help="Max number of angles to bunch at each energy.")
    parser.add_argument("-G", "--GroupAngles", type=int, default="1",  help="Number of energy batches for T2B transforms, aka batches")
    parser.add_argument("-a", "--anglesData", type=int, help="Max number of angular data points to use (to make smaller search). Pos: random selection. Neg: first block")
    parser.add_argument("-m", "--maxData", type=int, help="Max number of data points to use (to make smaller search). Pos: random selection. Neg: first block")

    parser.add_argument(      "--Large", type=float, default="40",  help="'large' threshold for parameter progress plotts.")
    parser.add_argument("-C", "--Cross_Sections", action="store_true", help="Output fit and data files for grace")
    parser.add_argument("-M", "--Matplot", action="store_true", help="Matplotlib data in .json output files")
    parser.add_argument("-l", "--logs", type=str, default='', help="none, x, y or xy for plots")
    parser.add_argument(      "--datasize", type=float,  metavar="size", default="0.2", help="Font size for experiment symbols. Default=0.2")
    parser.add_argument("-t", "--tag", type=str, default='', help="Tag identifier for this run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debugging output (more than verbose)")
    args = parser.parse_args()

    gnd=reactionSuiteModule.readXML(args.inFile)
    
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
    

    f = open( args.data )
    data_lines = f.readlines( )
    n_data = len(data_lines)
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
    if args.debug and False: 
        with open(args.data+'-/T/sorted','w') as fout: fout.writelines(data_lines)
    
    n_data = len(data_lines)
    data_val = numpy.zeros([n_data,5], dtype=DBLE)    # Elab,mu, datum,absError
    data_p   = numpy.zeros([n_data,2], dtype=INT)    # pin,pout
    
    if args.AngleBunching > 1:
        Energies = {}
        Uses = {}
        for l in data_lines:
            Ein = l.split()[0]
            count = Energies.get(Ein,0) + 1
            Energies[Ein] = count
            Uses[count] = Uses.get(count,0) + 1
            if count>1: Uses[count-1] -= 1

        print('Data points:',n_data,'with',len(Energies),'independent incident energies')
        maxcount = sorted(Uses.keys())[-1]
        print('Bunching:')
        for count in range(1,maxcount+1):
            Es = []
            for Ein in Energies.keys():
                if Energies[Ein] == count:  Es.append(Ein)
            print('%4i  %4i' % (count,Uses.get(count,0)) ) #, len(Es),  Es )
        print('---')

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
    f = open( args.norm )
    norm_lines = f.readlines( )
    f.close( )    
    n_norms= len(norm_lines)
    norm_val = numpy.zeros(n_norms, dtype=DBLE)  # norm,step,expect,syserror
    norm_info = numpy.zeros([n_norms,2], dtype=DBLE)  # norm,step,expect,syserror
    norm_refs= []    
    ni = 0
    n_cnorms = 0
    for l in norm_lines:
        parts = l.split()
#         print(parts)
        norm,step, name,expect,syserror,reffile = parts
        norm,step,expect,syserror = float(norm),float(step),float(expect),float(syserror)
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
        '\nData groups:',len(groups),'\nX4 groups:',len(X4groups),'\nVariable norms:',n_norms,' of which constrained:',n_cnorms)
    
    effect_norm = numpy.zeros([n_norms,n_data])
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

    finalStyleName = 'fitted'
    fitStyle = stylesModule.crossSectionReconstructed( finalStyleName,
            derivedFrom=gnd.styles.getEvaluatedStyle().label )

    print("Finish setup: ",tim.toString( ))
    base = args.inFile.replace('xml','')
    base += '+%s' % args.data.replace('.data','')
    if args.maxData    is not None: base += '_m%s' % args.maxData
    if args.anglesData is not None: base += '_a%s' % args.anglesData
    if args.Search     is not None: base += '+S' 
    if args.Iterations is not None: base += '_I%s' % args.Iterations
    dataDir = base
    os.system('mkdir '+dataDir)
    chisqtot,xsc,norm_val,n_pars = Rflow(gnd,partitions,base,data_val,data_p,n_angles,n_angle_integrals,Ein_list,args.Fixed,
                        norm_val,norm_info,norm_refs,effect_norm, args.LMatrix,args.GroupAngles,
                        args.Search,args.Iterations,args.restarts,args.Distant,args.Background,args.ReichMoore,  
                        args.verbose,args.debug,args.inFile,fitStyle,'_'+args.tag,args.Large)

    print("Finish rflow call: ",tim.toString( ))
    chisqPN = chisqtot / n_data
    print('\n ChiSq/pt = %10.4f from %i points' % (chisqPN,n_data))
    
    if args.Search or True:  
        print('Revised norms:',norm_val)
        docLines = ['Rflow:']
        for n in range(n_norms):
            docLines.append("&variable name='%s' kind=5 dataset=0, 0 datanorm=%f step=0.01, reffile='%s'/" % ( norm_refs[n][0] ,norm_val[n], norm_refs[n][1]) ) 
        docLines.append('\n')
        docLines.append('\n'.join(docData) )
#         print('docLines:',docLines)

        if previousFit:
            deck = 'Fitted_data'
            deckLabels = [item.keyValue for item in computerCodeFit.inputDecks]
            for i in range(2,100): 
                deckLabel = '%s %s' % (deck,i)
                if deckLabel not in deckLabels: break
            print('\nNew InputDeck is "%s" after' % deckLabel,deckLabels,'\n')
        else: 
            computerCodeFit = computerCodeModule.ComputerCode( label = 'R-matrix fit', name = 'Rflow', version = '', date = time.ctime() )
            
        inputDataSpecs = computerCodeModule.InputDeck( deckLabel , ('\n  %s\n' % time.ctime() )  + ('\n'.join( docLines ))+'\n' )
        computerCodeFit.inputDecks.add( inputDataSpecs )

        if not previousFit: RMatrix.documentation.computerCodes.add( computerCodeFit )

        info = '+S_' + args.tag
        open( base.replace('.xml','') + args.tag + '-fit.xml', mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )
    else:
        info = ''

    ngraphAll = 0
    groups = sorted(groups)
    chisqAll = 0
#     plot_cmds = []
    plot_cmd = 'xmgr '
    for group in groups:
        if args.Cross_Sections:
            g_out = group+info+'-fit'
            if '/' in g_out: g_out = dataDir + '/' + g_out.split('/')[1].replace('/','+')
            gf = open(g_out,'w')
            e_out = group+info+'-expt'
            if '/' in e_out: e_out = dataDir + '/' + e_out.split('/')[1].replace('/','+')
            ef = open(e_out,'w')
            op = ' in file %-43s' %  g_out
        else:
            op = ' in group %-43s' %  group
        ie = 0
        io = 0
        chisq = 0.0
        for id in range(n_data):
            if group == group_list[id]:
                fac = 1.0
                if not args.norm1:
                    for ni in range(n_norms):
                        fac += (norm_val[ni]-1.) * effect_norm[ni,id]
                Data = data_val[id,2]*fac
                DataErr = data_val[id,3]*fac  
                ex2cm = data_val[id,4] 
                chi = (xsc[id]/fac/ex2cm-data_val[id,2])/data_val[id,3] 
                
                if args.Cross_Sections:
                    cluster = cluster_list[id]
                    Ein = Ein_list[id]
                    Aex = Aex_list[id]
                    if cluster == 'A':
#                         theta = math.acos(data_val[id,1])*180./pi
                        print(Aex, xsc[ie]/ex2cm, chi, file=gf)
                        print(Aex, Data, DataErr, file=ef)                   
                    elif cluster in ['E','I']:
                        print(Ein, xsc[ie]/ex2cm, chi, file=gf)
                        print(Ein, Data, DataErr, file=ef)                                
                    else:  # cluster == 'N':  xyz
#                         theta = math.acos(data_val[id,1])*180./pi
                        print(Ein, Aex, xsc[ie]/ex2cm, chi, file=gf)   # xyz+chi
                        print(Ein, Aex, Data, DataErr, file=ef)  #xyzdz 
                        
                chisq += chi**2
                io += 1
            ie += 1
        if args.Cross_Sections:
            gf.close()
            ef.close()
        print('Model %2i curve (%4i pts)%s:   chisq/gp =%9.3f  %8.3f %%' % (ngraphAll+1,io,op,chisq/io,chisq/chisqtot*100.) )
        if args.Cross_Sections: 
            plot_cmd += ' -graph %i -xy %s -xydy %s ' % (ngraphAll,g_out,e_out) 
#             plot_cmds.append(plot_cmd)
        chisqAll += chisq
        ngraphAll += 1
#     plot_cmds.append(plot_cmd)
# chi from norm_vals themselves:
    for ni in range(n_norms):
        chi = (norm_val[ni] - norm_info[ni,0]) * norm_info[ni,1]        
        print('Norm scale   %10.6f         %-30s ~ %10.5f :    chisq    =%9.3f  %8.3f %%' % (norm_val[ni] , norm_refs[ni][0],norm_info[ni,0], chi**2, chi**2/chisqtot*100.) )
        chisqAll += chi**2
    print('\n Last chisq/pt  = %10.5f from %i points' % (chisqAll/n_data,n_data) )  
    dof = n_data + n_cnorms - n_norms - n_pars
    print(  ' Last chisq/dof = %10.5f' % (chisqAll/dof), '(dof =',dof,')\n' )   
    if args.Cross_Sections and ngraphAll < 20: 
         print("Plot with\n",plot_cmd,'\n with required rows and cols for',ngraphAll,'graphs')
    
#     for plot_cmd in plot_cmds: print("Plot:    ",plot_cmd)


    
    X4groups = sorted(X4groups)
    print('\nX4groups sorted:',X4groups)
#     if len(X4groups)<1: sys.exit()
    print('Data grouped by X4 subentry:')
    chisqAll = 0
    plot_cmds = []
    legendsize = 1.0 # 0.5 # default
    evaluation = gnd.evaluation 
    info = info.replace('_','')
    
    for group in X4groups:
        groupB = group.split('@')[0]
        ng = 0
        plot_cmd = ''
        ngraphAll = 0
        if args.Cross_Sections:
            g_out = group+info+'-fit'
            if '/' in g_out: g_out = dataDir + '/' + g_out.split('/')[1].replace('/','+')
            gf = open(g_out,'w')
            e_out = group+info+'-expt'
            if '/' in e_out: e_out = dataDir + '/' + e_out.split('/')[1].replace('/','+')
            ef = open(e_out,'w')
            op = ' in file %-43s' %  g_out
        else:
            op = ' in group %-43s' %  group

        io = 0
        chisq = 0.0
        lfac = 1.0
        GraphList = []
        DataLines = []
        ModelLines = []
        ng = 0
        
        curves = set()
        ptsInCurve = {}  # to be list of curves for given group base (groupB)
        for id in range(n_data):
            gld = group_list[id]
#             print('For data pt',id,'group=',group,'gld=',gld)
            glds= gld.split('@')
            if groupB == glds[0]:
                curve = glds[1] if len(glds)>1 else 'Aint'
                ptsInCurve[curve] = ptsInCurve.get(curve,0) + 1
                
                pin,pout = data_p[id,:]   # assume same for all data(id) in this curve!
                if pout == -1:
                    reaction = 'total'
                elif pout == pin:
                    reaction = 'elastic'
                else:
                    reaction = pins[pin]+'->'+pins[pout]
                    
                fac = 1.0 # assume same for all data(id) in this curve!
                if not args.norm1:
                    for ni in range(n_norms):
                        fac += (norm_val[ni]-1.) * effect_norm[ni,id]
                lfac = (fac-1)*100
                curves.add((curve,fac,lfac,pin,pout,reaction))
     
        if args.debug: print('\nGroup',group,'has curves:',curves)
        ncurve = 0
        for curve,fac,lfac,pin,pout,reaction in curves:
            tag = groupB + ( ('@' + curve) if curve!='Aint' else '')
            if ptsInCurve[curve]==0: continue
#             print('\nCurve for ',tag)
            
            lchisq = 0.
            if args.Matplot:
                LineData  = [{}, [[],[],[],[]] ]
                LineModel = [{}, [[],[],[],[]] ]

            for id in range(n_data):
                gld = group_list[id]
                if gld != tag: continue
     
#                 pin,pout = data_p[id,:]
#                 reaction = pins[pin]+('->'+pins[pout] if pout!=pin else ' elastic')
                Data = data_val[id,2]*fac
                DataErr = data_val[id,3]*fac        
                ex2cm = data_val[id,4] 
                chi = (xsc[id]/fac/ex2cm-data_val[id,2])/data_val[id,3] 
                
                cluster = cluster_list[id]
                Ein = Ein_list[id]
                Aex = Aex_list[id]
                if args.Cross_Sections:
                    if cluster == 'A':
#                         theta = math.acos(data_val[id,1])*180./pi
                        print(Aex, xsc[id]/ex2cm, chi, file=gf)
                        print(Aex, Data, DataErr, file=ef)                   
                    elif cluster in ['E','I']:
                        print(Ein, xsc[id]/ex2cm, chi, file=gf)
                        print(Ein, Data, DataErr, file=ef)                                
                    else:  # cluster == 'N':  xyz
#                         theta = math.acos(data_val[id,1])*180./pi
                        print(Ein, Aex, xsc[id]/ex2cm, chi, file=gf)   # xyz+chi
                        print(Ein, Aex, Data, DataErr, file=ef)  #xyzdz 
                if args.Matplot:    
                    xplot = float(Ein if cluster in ['E','I'] else Aex)
                    LineModel[1][0].append(xplot)
                    LineModel[1][1].append(xsc[id]/ex2cm)
                    LineModel[1][2].append(0)
                    LineModel[1][3].append(0)
                    LineData[1][0].append(xplot)
                    LineData[1][1].append(Data)
                    LineData[1][2].append(0)
                    LineData[1][3].append(DataErr)
                    
                chisq += chi**2
                lchisq += chi**2
                io += 1
            if args.Cross_Sections:   # end of curve
                print('&', file=gf)
                print('&', file=ef)
                
            if args.Matplot:    # end of curve
                if len(LineModel[1][0])==1:  # extend model line +-5% if only 1 data points
                    LineModel[1][0].append(LineModel[1][0][0]*0.99); LineModel[1][0].append(LineModel[1][0][0]*1.01)
                    LineModel[1][1].append(LineModel[1][1][0]);      LineModel[1][1].append(LineModel[1][1][0])
                    LineModel[1][2].append(0); LineModel[1][2].append(0)
                    LineModel[1][3].append(0); LineModel[1][3].append(0)
                    
                ng += 1
                ic = (ng-1) % 15 +1  # for colors
                leg = curve
#                         if type(leg) is list and len(leg)>1: leg=leg[1]
                legtag = leg if len(leg) < 8 else leg[:8]
                legend = legtag + '  X2/pt=%.2f' % (lchisq/ptsInCurve[curve])
                if lfac!=0.0: legend += ' n%s%.2f%%' % ('+' if lfac>0 else '-', abs(lfac))
                LineData[0] =  {'kind':'Data',  'color':plcolor[ic-1], 'capsize':0.10, 'legend':legend, 'legendsize':legendsize,
                    'symbol': plsymbol[ng%7], 'symbolsize':args.datasize   }
                LineModel[0] = {'kind':'Model', 'color':plcolor[ic-1], 'linestyle': pldashes[(ng-1)%4], 'evaluation':''} # evaluation.split()[0] }
#                 DataLines.append(LineData)
#                 ModelLines.append(LineModel)
                if args.debug: print('Finishing new curve',ng,'for',curve,'with legend',legend,'with',ptsInCurve[curve],'pts')

                DataLines.append(LineData)
                ModelLines.append(LineModel)
        
        ncurve += 1
        print('\nModel %2i %2i curves (%4i pts)%s:   chisq/gp =%9.3f  %8.3f %%' % (ncurve,ng,io,op,chisq/io,chisq/chisqtot*100.) )
        chisqAll += chisq
        ngraphAll += 1
        
        if args.Cross_Sections:    # wrap up this subentry
            gf.close()
            ef.close()
            plot_cmd += 'xmgr -xy %s -xydy %s ' % (g_out,e_out) 
            plot_cmds.append(plot_cmd)

        if args.Matplot:           # wrap up this subentry
            subtitle = "Using " + args.inFile + ' with  '+args.data+" & "+args.norm + ', Chisq/pt =%.3f' % (chisq/io)
            kind     = 'R-matrix fit of '+group.split('@')[0]+' for '+reaction+' (units mb and MeV)'
            GraphList.append([DataLines+ModelLines,subtitle,args.logs,kind])

            j_out = group+info+'.json'
            if '/' in j_out: j_out = dataDir + '/' + j_out.split('/')[1].replace('/','+')
            with open(j_out,'w') as ofile:
               json.dump([1,1,cmd,GraphList],ofile)
               
            plot_cmd += '\t             json2pyplot.py -w 10,8 %s' % j_out
            plot_cmds.append(plot_cmd)

# chi from norm_vals themselves:
    for ni in range(n_norms):
        chi = (norm_val[ni] - norm_info[ni,0]) * norm_info[ni,1]        
        print('Norm scale   %10.6f         %-30s ~ %10.5f :     chisq    =%9.3f  %8.3f %%' % (norm_val[ni] , norm_refs[ni][0],norm_info[ni,0], chi**2, chi**2/chisqtot*100.) )
        chisqAll += chi**2
    print('\n Last chisq/pt  = %10.5f from %i points' % (chisqAll/n_data,n_data) )  
    dof = n_data + n_cnorms - n_norms - n_pars
    print(  ' Last chisq/dof = %10.5f' % (chisqAll/dof), '(dof =',dof,')' )   
    
    for plot_cmd in plot_cmds: print("Plot:    ",plot_cmd)

print("Finish rflow: ",tim.toString( ))
