#!/usr/bin/env python3

##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

import math,numpy,cmath

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
pi = 3.1415926536
rsqr4pi = 1.0/(4*pi)**0.5


def potl(r, zz, OpticalParameters):
    V,RZZ,AZ, W,RWZ,AW, WD,RDZ,AD, VSO,RSOZ,ASO, RC = OpticalParameters
    Cshape = (1.5 - 0.5 * (r/RC)**2)/RC if r < RC else 1./r 
    VC = Cshape * coulcn * zz
    Vvol = - V / (1 + math.exp( (r - RZZ)/AZ ) )
    Wvol = - W / (1 + math.exp( (r - RWZ)/AW ) )
    ED =  math.exp( (r - RDZ)/AD )
    Wsrf = - WD  * 4.0 * ED / (1 + ED)**2
    Vopt = complex(Vvol + VC, Wvol+Wsrf)
#    print(r, Vvol,Wvol,Wsrf, file=open('potentials','a')) 
    return(Vopt)

def KoningDelaRoche(proj,xZ,xA,E):  # for n and H1
    xN=xA-xZ
    diff = (xN - xZ)/xA
    energy = E
    AOV = xA**0.333333333e0
    if proj=='H1':
        v1 = 59.30e0 + 21.0e0*diff - 2.4e-2*xA
        v2 = 7.067e-3 + 4.23e-6*xA
        v3 = 1.729e-5 - 1.136e-8*xA
        v4 = 7.1e-9
        w1 = 14.667e0 + 9.629e-3*xA
        w2 = 73.55e0 + 7.95e-2*xA
        d1 = 16.0e0 + 16.0e0*diff
        d2 = 1.80e-2 + 3.802e-3/(1.0e0 + math.exp((xA - 156.0e0)/8.0e0))
        d3 = 11.5e0
        vso_1 = 5.922e0 + 3.0e-3*xA
        vso_2 = 4.0e-3
        wso_1 = -3.1e0
        wso_2 = 160.0e0
        eFermi = -8.4075e0 + 1.378e-2*xA
        rc = 1.198e0 + 0.697e0/(xA**(0.6666666666e0)) + 12.994e0/(xA**(1.6666666666e0))
        vc = 1.73e0*xZ/(rc*(xA**0.333333333e0))
    elif proj=='n':
        v1 = 59.30e0 - 21.0e0*diff - 2.4e-2*xA
        v2 = 7.228e-3 - 1.48e-6*xA
        v3 = 1.994e-5 - 2.0e-8*xA
        v4 = 7.1e-9
        w1 = 12.195e0 + 1.67e-2*xA
        w2 = 73.55e0 + 7.95e-2*xA
        d1 = 16.0e0 - 16.0e0*diff
        d2 = 1.80e-2 + 3.802e-3/(1.0e0 + math.exp((xA - 156.0e0)/8.0e0))
        d3 = 11.5
        vso_1 = 5.922e0 + 3.0e-3*xA
        vso_2 = 4.0e-3
        wso_1 = -3.1e0
        wso_2 = 160.0e0
        eFermi = -11.2814e0 + 2.646e-2*xA
        rc = 0.0e0
        rc = 1.198e0 + 0.697e0/(xA**(0.6666666666e0)) + 12.994e0/(xA**(1.6666666666e0))
        vc = 0.0e0
    else:
        print("KoningDelaRoche potential not suitable for ",proj,' only H1 or n')
        sys.exit()
        
    del_vc = vc*v1*(v2 - 2.0e0*v3*(energy - eFermi) + 3.0e0*v4*(energy - eFermi)**2)

    V = v1*(1.0e0 - v2*(energy - eFermi) + v3*(energy - eFermi)**2 - v4*(energy - eFermi)**3) + del_vc
    RZZ = 1.3039e0 - 0.4054e0/(xA**0.333333333e0)
    AZ = 0.6778e0 - 1.487e-4*xA

    W = w1*(energy - eFermi)**2/((energy - eFermi)**2 + w2**2)
    RWZ = RZZ
    AW = AZ

    WD = d1*(energy - eFermi)**2*math.exp(-d2*(energy - eFermi))/((energy - eFermi)**2 + d3**2)
    RDZ = 1.3424e0 - 1.585e-2*(xA**0.333333333e0)
    if proj=='n': AD = 0.5446e0 - 1.656e-4*xA     
    if proj=='H1':  AD = 0.5187e0 - 5.206e-4*xA

    WV = 0.0e0
    RWV = 1.3424e0 - 1.585e-2*(xA**0.333333333e0)
    AWV = AD

    VSO = vso_1*math.exp(-vso_2*(energy - eFermi))
    RSOZ = 1.1854e0 - 0.647e0/(xA**0.333333333e0)      
    ASO = 0.59e0

    WSOI = wso_1*(energy - eFermi)**2/((energy - eFermi)**2 + wso_2**2)
    RSOI = RSOZ
    ASOI = ASO
    return( (V,RZZ*AOV,AZ, W,RWZ*AOV,AW, WD,RDZ*AOV,AD, VSO,RSOZ*AOV,ASO, rc) )

def Soukhovitskii(proj,xZ,xA,E):  # for n and H1
    print('Soukhovitskii not implemented yet')
    sys.exit()
    return

def becchetti(proj,Z,A,E):  # for n and H1
    AOV=A**0.333333333
    RC = 1.3 * AOV
    N=A-Z
    if proj=='H1':
        V=54.0-0.32*E+24.*(N-Z)/A+0.4*Z/AOV
        RZ=1.17
        RZZ=RZ*AOV
        AZ=0.75
        W=0.22*E-2.7
        if W < 0.00001: W=0.
        WD=11.8-0.25*E+12.*(N-Z)/A
        if WD < 0.00001: WD=0.
        RW=1.32
        RWZ=RW*AOV
        RD=RW
        RDZ=RWZ
        AW=0.51+0.7*(N-Z)/A
        AD=AW
        VSO=6.2
        RSO=1.01
        RSOZ=RSO*AOV
        ASO=0.75
#         FJ=4.1887902*V*RZZ**3*(1+9.8696044*AZ**2/RZZ**2)/A

    elif proj=='n':
        V=56.3-0.32*E-24.0*(N-Z)/A
        RZ=1.17
        RZZ=RZ*AOV
        AZ=0.75
        W=0.22*E-1.56
        if W < 0.00001: W=0.
        WD=13.0-0.25*E-12.*(N-Z)/A
        if WD < 0.00001: WD=0.
        RW=1.26
        RWZ=RW*AOV
        RD=RW
        RDZ=RWZ
        AW=0.58
        AD=AW
        VSO=6.2
        RSO=1.01
        RSOZ=RSO*AOV
        ASO=0.75
#         FJ=4.1887902*V*RZZ**3*(1+9.8696044*AZ**2/RZZ**2)/A
    else:
        print("Becchetti potential not suitable for ",proj,' only H1 or n')
        sys.exit()

    return( (V,RZZ,AZ, W,RWZ,AW, WD,RDZ,AD, VSO,RSOZ,ASO, RC) )

def islam(proj,Z,A,E):  # for n on 14N
    AOV=A**0.333333333
    RC = 1.3 * AOV
    N=A-Z
    if proj=='n':
        V=52.4-0.29*E
        R0=1.197
        R=R0*AOV
        A=0.593
        
        W=0.267*E - 5.86 if E > 22 else 0. 
        WD=5.43 - 0.16*(22-E) if E<=22 else 8.64 - 0.145*E
        RW0=1.388
        RW=RW0*AOV
        AW=0.593
        
        VSO=6.0
        RSO=1.01
        RSOZ=RSO*AOV
        ASO=0.30
    else:
        print("Islam potential not suitable for ",proj,' only n')
        sys.exit()

    return( (V,R,A, W,RW,AW, WD,RW,AW, VSO,RSOZ,ASO, RC) )

def perey_d(proj,Z,A,E):    # for H2
    if proj != 'H2':
        print("Perey_d potential not suitable for ",proj,' only H2')
        sys.exit()
            
    xA13=A**0.333333333
    RC = 1.15 * xA13

    V =  81.0 - 0.22*E + Z/xA13
    R = 1.15 * xA13
    AV = 0.81
    
    WD = 14.4 + 0.24*E
    RD = 1.34 * xA13
    AD = 0.68
    return( (V,R,AV, 0.0,R,AV, WD,RD,AD,  0.0,R,AV, RC) )


def perey_t(proj,Z,A,E):    # for H3
#    Quoted in C. M. Perey and F. G. Perey, At. Data and Nuc. Data Tables,
#    Vol. 17, No. 1, 1976, p. 1.

    if proj != 'H3':
        print("Perey_d potential not suitable for ",proj,' only H3')
        sys.exit()
            
    xA13=A**0.333333333
    RC = 1.15 * xA13
    N = A - Z
    diff = (N - Z)/A

    V =  165.0 - 0.17*E - 6.4*diff
    R = 1.20 * xA13
    AV = 0.72
    
    W = 46.0 - 0.33*E - 110.0*diff
    RW = 1.40 * xA13
    AW = 0.84
    
    
    VSO=2.5
    RSOZ=1.20 * xA13
    ASO=0.72
    
    return( (V,R,AV, W,RW,AW, 0.0,RW,AW,  VSO,R,AV, RC) )
    
def Li_t(proj,Z,A,E):    # for H3
#    X. Li et al Global potential Energy < 40 MeV',
#    NPA 789 (2007) 103

    if proj != 'H3':
        print("Li_t potential not suitable for ",proj,' only H3')
        sys.exit()
            
    a13=A**0.333333333
    N = A - Z
    diff = (N - Z)/A
    
    ed = E
    ee = E
    e  = Z
    an = diff

    V =  137.6-0.1456*ed+0.0436*ed*ed+4.3751*an + 1.0474*e/a13
    R = 1.12010-0.1504/a13
    AV = 0.68330+0.01910*a13
    
    W = 7.383+0.5025*ee-0.0097*ed*ed
    RW =1.32020-0.1776/a13
    AW = 1.11900+0.01913*a13

    WD = 37.06-0.6451*ee-47.19*an
    RD = 1.25100-0.4622/a13
    AD = 0.81140+0.01159*a13
        
    VSO = 1.9029
    RSOZ = 0.46991+0.1294/a13
    ASO =0.35450-0.05220*a13

    RC = 1.15 
        
    return( (V,R*a13,AV, W,RW*a13,AW, WD,RD*a13,AD,  VSO,RSOZ*a13,ASO, RC*a13) )
       
    
    
def perey_h(proj,Z,A,E):    # for He3
#    Quoted in C. M. Perey and F. G. Perey, At. Data and Nuc. Data Tables,
#    Vol. 17, No. 1, 1976, p. 1.

    if proj != 'He3':
        print("Perey_d potential not suitable for ",proj,' only He3')
        sys.exit()
            
    xA13=A**0.333333333
    RC = 1.15 * xA13
    N = A - Z
    diff = (N - Z)/A

    V =  151.9 - 0.17*E - 50.*diff
    R = 1.20 * xA13
    AV = 0.72
    
    W = 41.7 - 0.33*E + 44.0*diff
    RW = 1.40 * xA13
    AW = 0.88
    
    
    VSO=2.5
    RSOZ=1.20 * xA13
    ASO=0.72
    
    return( (V,R,AV, W,RW,AW, 0.0,RW,AW,  VSO,R,AV, RC) )
            
def Avrigeanu(proj,Z,A,E):  # for He4
    if proj != 'He4':
        print("Avrigeanu potential not suitable for ",proj,' only He4')
        sys.exit()
            
    xA13=A**0.333333333
    RC = 1.245 * xA13
    
    V = 101.1 - 0.248*E + 6.051*Z/xA13
    R = 1.20 * xA13
    AV = 0.72
    
    if E < 73.0:
        W = 12.64 + 0.2  *E - 1.706*xA13
    else:
        W = 26.82 + 0.006*E - 1.706*xA13
    RW = 1.57 * xA13
    AW = 0.692 - 0.02*xA13
    
    return( (V,R,AV, W,RW,AW, 0.0,RW,AW, 0.0,R,AV, RC) )


def get_optical_S(sc_info,n, ompout):

    nsc = len(sc_info)
    g = numpy.zeros([nsc,n+1], dtype=CMPLX)
    f = numpy.zeros([nsc,n+1], dtype=CMPLX)
    wf  = numpy.zeros([nsc,n+1], dtype=CMPLX)
    Lc  = numpy.zeros([nsc], dtype=CMPLX)
    k  = numpy.zeros(nsc, dtype=REAL)
    hcm  = numpy.zeros([nsc,1], dtype=REAL)
    phis  = numpy.zeros([nsc], dtype=REAL)
    ar  = numpy.zeros([nsc], dtype=REAL)
    Pspacing  = numpy.zeros([nsc], dtype=REAL)
    
    isc = 0
    optopt = {}
    optparams = []
    for jset,c,p,h,L,Spin,pair,E,Pspace,a,rmass,pname,ZP,ZT,AT,L_coul,phi, OpticalPot in sc_info:
        coef =  -1.0 / (fmscal * rmass)
        k[isc] = math.sqrt( fmscal * rmass * E)
        Lc[isc] = complex(L_coul[0], L_coul[1])
        hcm[isc,0] = h
        ar[isc] = a
        phis[isc] = phi 
        Elab = E * AT / (AT - rmass)
        Pspacing[isc] = Pspace
        
        if pname=='n':
            if OpticalPot[0] <= 1: 
                OpticalParameters = KoningDelaRoche(pname,ZT,AT,Elab)
                opt = 'n: KoningDelaRoche'
            if OpticalPot[0] == 2: 
                OpticalParameters = Soukhovitskii(pname,ZT,AT,Elab)
                opt = 'n: Soukhovitskii'
            if OpticalPot[0] == 3: 
                OpticalParameters = becchetti(pname,ZT,AT,Elab)
                opt = 'n: Becchetti'
            if OpticalPot[0] >= 4: 
                OpticalParameters = islam(pname,ZT,AT,Elab)
                opt = 'n: Islam'
        elif pname=='H1':
            if OpticalPot[0] <= 1: 
                OpticalParameters = KoningDelaRoche(pname,ZT,AT,Elab)
                opt = 'p: KoningDelaRoche'
            if OpticalPot[0] == 2: 
                OpticalParameters = Soukhovitskii(pname,ZT,AT,Elab)
                opt = 'p: Soukhovitskii'
            if OpticalPot[0] >= 3: 
                OpticalParameters = becchetti(pname,ZT,AT,Elab)
                opt = 'p: Becchetti'
        elif pname=='H2':
            OpticalParameters = perey_d(pname,ZT,AT,Elab)
            opt = 'd: Perey'
        elif pname=='H3':
            if OpticalPot[0] <= 1: 
                OpticalParameters = perey_t(pname,ZT,AT,Elab)
                opt = 't: Perey'
            if OpticalPot[0] >= 2: 
                OpticalParameters = Li_t(pname,ZT,AT,Elab)
                opt = 't: Li'
        elif pname=='He3':
            OpticalParameters = perey_h(pname,ZT,AT,Elab)
            opt = 'h: Perey'
        elif pname=='He4':
            OpticalParameters = Avrigeanu(pname,ZT,AT,Elab)        
            opt = 'a: Avrigeanu'
        else:
            print('Unrecognized projectile',pname)
            sys.exit()
        
        optparams.append(OpticalParameters)
        opop = optopt.get(pname,None)
        if opop is None: print('Optical potential for',pname,'is',opt)
        optopt[pname] = opt
        
        for i in range(1,n+1):
            r = i*h
            potls = potl(r,ZP*ZT,OpticalParameters) - coef * L*(L+1)/(r*r)
            g[isc,i] = (potls - E) / coef
            
        wf[isc,0] = 0.0
        wf[isc,1] = (k[isc]*h)**(L+1)
        f[isc,:] = 1. +  hcm[isc,0]**2/12 * g[isc,:]
        isc += 1
            
#     f[:,:] = 1. +  hcm[:,0]**2/12 * g[:,:]
    for i in range(1,n):
        wf[:,i+1] = ( ( 12.-10.*f[:,i] ) * wf[:,i] - f[:,i-1] * wf[:,i-1] ) / f[:,i+1]
    
    deriv = (147.0*wf[:,n]-360.0*wf[:,n-1]+450.0*wf[:,n-2]  -400.0*wf[:,n-3]+225.0*wf[:,n-4]-72.*wf[:,n-5]+10.*wf[:,n-6] ) /(60.*hcm[:,0])
    Rmat = wf[:,n] / (ar*deriv)
    Smat   = (1. - Rmat * numpy.conjugate(Lc[:]) )/ (1. - Rmat * Lc[:])  * numpy.exp(complex(0.,2.)*phis[:])
    TC = 1. - (Smat * numpy.conjugate(Smat)).real
    delta = numpy.real( numpy.log(Smat)/complex(0.,2.)) * 180./pi
#    return(Smat)
    
    isc = 0
    for jset,c,p,h,L,Spin,pair,E,Pspace,a,rmass,pname,ZP,ZT,AT,L_coul,phi, OpticalPot in sc_info:
        print('    %5i is p%2i, LS=%i,%s, Ecm %8.3f, delta %9.2f, TC = %9.5f' % (isc,pair,L,Spin,E,delta[isc], TC[isc] ), ) #,phis[isc]*180/pi),-phis[isc]/(a* k[isc]) ) #, Smat[isc], TC[isc] , file=ompout)
#       print(E,TC[isc],    file = open('trans%i' % L,'a') )
#       print(E,optparams[isc],    file = open('opticalp%i' % L,'a') )
        isc += 1
    return(Smat, Pspacing)
    
if __name__=="__main__":
    import sys
    print('Arguments projectile name (GNDS form), target Z,A, energy:')
    pname,ZT,AT,Elab,Version = sys.argv[1:6]
    ZT,AT,Elab,Version = int(ZT),int(AT),float(Elab),int(Version)
    
    if pname=='n':
        if Version <= 1: 
            OpticalParameters = KoningDelaRoche(pname,ZT,AT,Elab)
            opt = 'n: KoningDelaRoche'
        if Version == 2: 
            OpticalParameters = Soukhovitskii(pname,ZT,AT,Elab)
            opt = 'n: Soukhovitskii'
        if Version == 3: 
            OpticalParameters = becchetti(pname,ZT,AT,Elab)
            opt = 'n: Becchetti'
        if Version >= 4: 
            OpticalParameters = islam(pname,ZT,AT,Elab)
            opt = 'n: Islam'
            
    elif pname=='H1':
        if Version <= 1: 
            OpticalParameters = KoningDelaRoche(pname,ZT,AT,Elab)
            opt = 'p: KoningDelaRoche'
        if Version == 2: 
            OpticalParameters = Soukhovitskii(pname,ZT,AT,Elab)
            opt = 'p: Soukhovitskii'
        if Version >= 3: 
            OpticalParameters = becchetti(pname,ZT,AT,Elab)
            opt = 'p: Becchetti'

    elif pname=='H2':
        OpticalParameters = perey_d(pname,ZT,AT,Elab)
        opt = 'd: Perey'
    elif pname=='H3':
        if Version <= 1: 
            OpticalParameters = perey_t(pname,ZT,AT,Elab)
        opt = 't: Perey'
        if Version >= 2: 
            OpticalParameters = Li_t(pname,ZT,AT,Elab)
        opt = 't: Li'
    elif pname=='He3':
        OpticalParameters = perey_h(pname,ZT,AT,Elab)
        opt = 'h: Perey'
    elif pname=='He4':
        OpticalParameters = Avrigeanu(pname,ZT,AT,Elab)
        opt = 'a: Avrigeanu'
    else:
        print('Unrecognized projectile',pname)

    V,RZZ,AZ, W,RWZ,AW, WD,RDZ,AD, VSO,RSOZ,ASO, RC = OpticalParameters
    ZA = ZT*1000+AT
    AC = 0.0
    print('Optical potential for %s on ZA = %s at %7.3f MeV lab, version: %s' % (pname,ZA,Elab,opt))
    
    print('\nFresco line mode:')
    kpp = {'n':1, 'H1':1, 'H2':2, 'H3':3, 'He3':3, 'He4':4}
    kp = kpp.get(pname,1)
    print('%3i%2i 0 %8.4f%8.4f%8.4f%8.4f %s%s' % (kp,0,1.0,0.0,RC,AC,34*' ','A1,A2,RC'))
    print('%3i%2i 0 %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f %s%s' % (kp,1,V,RZZ,AZ,W,RWZ,AW,18*' ','V,R,A,W,RW,AW') )
    print('%3i%2i 0 %s%8.4f%8.4f%8.4f %s%s' % (kp,-2,24*' ',WD,RDZ,AD,18*' ','WD,RD,AD'))
    print('%3i%2i 0 %8.4f%8.4f%8.4f %s%s' % (kp,3,VSO,RSOZ,ASO,42*' ','VSO,RSO,ASO'))
    
    print('\nFresco namelist:')
    WSO,RSOZ,ASO = 0.,RSOZ,ASO
    print('&pot kp=%2i type=%2i p(1:3)=%8.4f%8.4f%8.4f%8.4f / ! %s + %s at %.3f MeV lab' % (kp,0,1.0,0.,RC,AC,pname,ZA,Elab) )
    print('&pot kp=%2i type=%2i p(1:6)=%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f /' % (kp,1,V,RZZ,AZ,W,RWZ,AW))
    print('&pot kp=%2i type=%2i p(1:6)=%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f /' % (kp,-2,0.,0.,0.,WD,RDZ,AD))
    print('&pot kp=%2i type=%2i p(1:6)=%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f /' % (kp,3,VSO,RSOZ,ASO,WSO,RSOZ,ASO))
