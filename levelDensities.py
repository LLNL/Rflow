
##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

import math,sys

from xData import XYs1d as XYs1dModule

pi = 3.1415926536

def leveldensity(jpi,LevelParms,emin,emax,A):

    J,parity  = jpi
    par = '+' if int(parity) > 0 else '-'
    print()
    model = LevelParms['Level-density model']
    aparam = LevelParms['a'] # 1
    spin_cut = LevelParms['spin_cut'] # 2
    delta = LevelParms['Delta'] # 3
    shell = LevelParms['Shell'] # 4
    gamma = LevelParms['gamma'] # 5
    ematch = LevelParms['E_match'] # 6
    ecut = LevelParms['E_cut'] # 7
    sg2cut = LevelParms['sigma_cut'] # 8
    low_e_mod = LevelParms['low_e_mod'] # 9
    rot_mode = int(LevelParms['rot_mode']+0.5) # 10
    vib_mode = int(LevelParms['vib_mode']+0.5) # 11s\
    sig = LevelParms['sig'] # 12
    sig_model = int(LevelParms['sig_model']+0.5) # 13
    T = LevelParms['T'] # 14
    E0 = LevelParms['E0']   # 15
    
    pfac = LevelParms['Parity Factor']
    K_vib = LevelParms['K_vib']
    K_rot = LevelParms['K_rot']
    
    dE = 0.1  #  
    nE = max(int( (emax - emin)/dE + 1.5), 0)
    dE = (emax-emin)/(nE-1) 
    
#     E = ematch
#     U = E - delta
#     sig2 = sig2_param(E,LevelParms,A)
#     apu = aparam_u(U,aparam,shell,gamma)
#     rhoFT = rho_FT(E, E0, T)
#     rhoBFM = rho_BFM(U,apu,low_e_mod,sig2)
#     print('At E,delta,U,A =',E,delta,U,A)
#     print('sig2, sig2**0.5 =',sig2,sig2**0.5)
#     print('aparam,apu,shell,gamma,E0,T',aparam,apu,shell,gamma,E0,T)
#     print('FT ::',rhoFT,'  BFM::',rhoBFM)
    
    
    density = [[emin-0.1,0.]]
    for iE in range(nE):
        E = emin + iE * dE 
        U = E - delta
        sig2 = sig2_param(E,LevelParms,A)
        apu = aparam_u(U,aparam,shell,gamma)
        
        if E < ematch:                     #  finite temperature
            rho = rho_FT(E, E0, T)
            K_rot = 1.0
            K_vib = 1.0

        else:                                   #  fermi Gas
            rho = rho_BFM(U,apu,low_e_mod,sig2)

#             K_rot = enhance_rot(rot_mode, sig2, rot_enhance, E)
#             K_vib = enhance_vib(vib_mode, A, U, E, apu, Shell, vib_enhance)

        enhance = K_rot*K_vib
        rho = rho*enhance
   
        jfac = spin_fac(J,sig2)
   
        pdf = rho * jfac * pfac
        density.append([E,pdf])

    densityFunction = XYs1dModule.XYs1d( data = density, dataForm = 'xys')
    Nlevels = densityFunction.integrate()
    Nlevels = float(Nlevels)
    print('For %.1f%s there are %.1f levels' % (J,par,Nlevels))
    
    cdf = XYs1dModule.XYs1d( data = [ densityFunction.domainGrid, densityFunction.runningIntegral( ) ], dataForm = 'xsandys' )
    for iE in range(nE):
        E = emin + iE * dE 
        pdf = densityFunction.evaluate(E)
        if iE % 5 == 0 and pdf > 0.25:
            print('E = %8.3f, J = %.1f%s, rho = %10.4f, cdf = %10.4f' % (E,J,par,pdf,cdf.evaluate(E)))
    inverse_cdf = cdf.inverse()

#     print('   pdf:',densityFunction.toString())
#     print('   cdf:',cdf.toString())
#     
#     print('   inverse_cdf:',inverse_cdf.toString())

    levelCandidates = []
    for i in range(int(Nlevels)):
        ei = inverse_cdf.evaluate(float(i)+0.5)
        levelCandidates.append(ei)
    print('Ep =',', '.join(['%.2f' %  ei for ei in levelCandidates]))
    return(levelCandidates, densityFunction)
    
def spin_fac(xJ,sg2):
#    This function returns the spin-dependence factor for the level density
    spin_fac = (2.*xJ+1.)*math.exp(-(xJ+0.5)**2/(2.*sg2))/(2.*sg2)
    return(spin_fac)

def sig2_param(E,LevelParms,A):

    aparam = LevelParms['a'] # 1
    spin_cut = LevelParms['spin_cut'] # 2
    delta = LevelParms['Delta'] # 3
    shell = LevelParms['Shell'] # 4
    gamma = LevelParms['gamma'] # 5
    ematch = LevelParms['E_match'] # 6
    ecut = LevelParms['E_cut'] # 7
    sg2cut = LevelParms['sigma_cut'] # 8
    low_e_mod = LevelParms['low_e_mod'] # 9
    rot_mode = int(LevelParms['rot_mode']+0.5) # 10
    vib_mode = int(LevelParms['vib_mode']+0.5) # 11
    sig = LevelParms['sig'] # 12
    sig_model = int(LevelParms['sig_model']+0.5) # 13
    
    U = E - delta
    Um = ematch - delta

    if E < ecut: return ( sg2cut )

    if E < ematch :
       apu = aparam_u(Um,aparam,shell,gamma)
       sig2_em = sig*math.sqrt(max(0.2,Um*apu))/aparam
       if sig_model == 0: sig2_em = sig*math.sqrt(max(0.2,Um*apu))/aparam
       if sig_model == 1: sig2_em = sig*math.sqrt(max(0.2,Um)/apu)
       sig2_em = max(sig2_em,(0.83*A**0.26)**2)
       deriv = (sig2_em - sg2cut)/(ematch - ecut)
       sig2 = sig2_em - deriv*(ematch - E)
       if sig2 < 0.0:
         print('sig2',sig2,' negative#')
         sys.exit()
    else:
       apu = aparam_u(U,aparam,shell,gamma)
       sig2 = (0.83*A**0.26)**2
       if U > 0.0:
          if sig_model == 0:
             sig2 = sig*math.sqrt(apu*U)/aparam
          elif sig_model == 1:
             sig2 = sig*math.sqrt(U/apu)
          sig2 = max(sig2,(0.83*A**0.26)**2)
#     print('E,sig_model,sig=',E,sig_model,sig,'sig2,s =',sig2,sig2**0.5)

    return ( sig2 )

def aparam_u(u,aparam,shell,gamma):
#    This function returns the excitation-energy dependent a-parameter

    if u >= 1.0e-6:
        aparam_u = aparam*(1.+shell*(1.-math.exp(-gamma*u))/u)
    else:
        aparam_u = aparam*(1.+shell*gamma)
    return(aparam_u ) 

def rho_FT(e,E0,T):
#    This subroutine returns the level density using the
#    finite-temperature model
    rho = math.exp((e-E0)/T)/T
    return ( rho )
    
def rho_BFM(U,apu,low_e_mod,sig2):
#    This subroutine returns the level density using the
#    back-shifted fermi gas model
    exponent1 = 2.0*math.sqrt(apu*U)
    U1 = U
    if U <= 0.: U1 = 1.0e-6
    rho_F = math.exp(exponent1)/(math.sqrt(2.0*sig2) * 12.0 * apu**0.25 * U1**1.25)
    
    
    
#     T = math.sqrt(U/apu)
#     an = apu/2.0
#     ap = apu/2.0
#     exponent2 = 4.0*ap*an*T**2
#     if int(low_e_mod) == 0:
#         rho = rho_F
#     elif  int(low_e_mod) == 1:
#         if U < 10.0:
#             rho = rho_F
#         else:
#             rho = rho_F
    return( rho_F )
