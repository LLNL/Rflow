
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
    Cshape = (1.5 - 0.5 * (r/RC)**2) if r < RC else 1./r 
    VC = Cshape * coulcn * zz
    Vvol = - V / (1 + math.exp( (r - RZZ)/AZ ) )
    Wvol = - W / (1 + math.exp( (r - RWZ)/AW ) )
    ED =  math.exp( (r - RDZ)/AD )
    Wsrf = - WD  * 4.0 * ED / (1 + ED)**2
    Vopt = complex(Vvol + VC, Wvol+Wsrf)
        
    return(Vopt)

def becchetti(proj,Z,A,E):  # for n and H1
    AOV=A**0.333333333
    RC = 1.3 * AOV
    FN=A-Z
    if proj=='H1':
        V=54.0-0.32*E+24.*(FN-Z)/A+0.4*Z/AOV
        RZ=1.17
        RZZ=RZ*AOV
        AZ=0.75
        W=0.22*E-2.7
        if W < 0.00001: W=0.
        WD=11.8-0.25*E+12.*(FN-Z)/A
        if WD < 0.00001: WD=0.
        RW=1.32
        RWZ=RW*AOV
        RD=RW
        RDZ=RWZ
        AW=0.51+0.7*(FN-Z)/A
        AD=AW
        VSO=6.2
        RSO=1.01
        RSOZ=RSO*AOV
        ASO=0.75
#         FJ=4.1887902*V*RZZ**3*(1+9.8696044*AZ**2/RZZ**2)/A

    elif proj=='n':
        V=56.3-0.32*E-24.0*(FN-Z)/A
        RZ=1.17
        RZZ=RZ*AOV
        AZ=0.75
        W=0.22*E-1.56
        if W < 0.00001: W=0.
        WD=13.0-0.25*E-12.*(FN-Z)/A
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
    FN = A - Z
    diff = (FN - Z)/A

    V =  165.0 - 0.17*E - 6.4*diff
    R = 1.20 * xA13
    AV = 0.72
    
    W = 46.0 - 0.33*E - 110.0*diff
    RW = 1.40 * xA13
    AW = 0.84
    
    
    VSO=2.5
    RSOZ=1.20*AOV
    ASO=0.72
    
    return( (V,R,AV, W,RW,AW, 0.0,RW,AW,  VSO,R,AV, RC) )
    
def perey_h(proj,Z,A,E):    # for He3
#    Quoted in C. M. Perey and F. G. Perey, At. Data and Nuc. Data Tables,
#    Vol. 17, No. 1, 1976, p. 1.

    if proj != 'He3':
        print("Perey_d potential not suitable for ",proj,' only He3')
        sys.exit()
            
    xA13=A**0.333333333
    RC = 1.15 * xA13
    FN = A - Z
    diff = (FN - Z)/A

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
    if E < 73.0:
        V = 12.64 + 0.2*E - 1.706*xA13
    else:
        V = 26.82 + 0.006*E - 1.706*xA13
    R = 1.57 * xA13
    AV = 0.692 - 0.02*xA13

    return( (V,R,AV, W,R,AV, 0.0,R,AV, 0.0,R,AV, RC) )


def get_optical_S(sc_info,n):

    nsc = len(sc_info)
    g = numpy.zeros([nsc,n+1], dtype=CMPLX)
    f = numpy.zeros([nsc,n+1], dtype=CMPLX)
    wf  = numpy.zeros([nsc,n+1], dtype=CMPLX)
    Lc  = numpy.zeros([nsc], dtype=CMPLX)
    k  = numpy.zeros(nsc, dtype=REAL)
    hcm  = numpy.zeros([nsc,1], dtype=REAL)
    phis  = numpy.zeros([nsc], dtype=REAL)
    ar  = numpy.zeros([nsc], dtype=REAL)
    
    isc = 0
    for jset,c,p,h,L,Spin,pair,E,a,rmass,pname,ZP,ZT,AT,L_coul,phi, OpticalPot in sc_info:
        coef =  -1.0 / (fmscal * rmass)
        k[isc] = math.sqrt( fmscal * rmass * E)
        Lc[isc] = complex(L_coul[0], L_coul[1])
        hcm[isc,0] = h
        ar[isc] = a
        phis[isc] = phi 
        
        if pname=='n':
            OpticalParameters = becchetti(pname,ZT,AT,E)
        elif pname=='H1':
            OpticalParameters = becchetti(pname,ZT,AT,E)
        elif pname=='H2':
            OpticalParameters = perey_d(pname,ZT,AT,E)
        elif pname=='H3':
            OpticalParameters = perey_t(pname,ZT,AT,E)
        elif pname=='He3':
            OpticalParameters = perey_h(pname,ZT,AT,E)
        elif pname=='He4':
            OpticalParameters = Avrigeanu(pname,ZT,AT,E)        
        else:
            print('Unrecognized projectile',pname)
            sys.exit()
                    
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
    
    firstL0 = True

    deriv = (147.0*wf[:,n]-360.0*wf[:,n-1]+450.0*wf[:,n-2]  -400.0*wf[:,n-3]+225.0*wf[:,n-4]-72.*wf[:,n-5]+10.*wf[:,n-6] ) /(60.*hcm[:,0])
    Rmat = wf[:,n] / (ar*deriv)
    Smat   = (1. - Rmat * numpy.conjugate(Lc[:]) )/ (1. - Rmat * Lc[:])  * numpy.exp(complex(0.,2.)*phis[:])
    TC = 1. - (Smat * numpy.conjugate(Smat)).real
    delta = numpy.real( numpy.log(Smat)/complex(0.,2.)) * 180./pi
    return(Smat)
    
    isc = 0
    for jset,c,p,h,L,Spin,pair,E,a,rmass,pname,ZP,ZT,AT,L_coul,phi, OpticalPot in sc_info:
        exd = k[isc] * a * math.cos( a* k[isc])
        rmd = math.sin( a * k[isc]) / exd
        print(isc,'is p%i, L=%i, E %8.3f, delta %9.2f, TC = %9.5f' % (pair,L,E,delta[isc], TC[isc] ) ) #,phis[isc]*180/pi),-phis[isc]/(a* k[isc]) ) #, Smat[isc], TC[isc] )
        isc += 1
    return(Smat)
    