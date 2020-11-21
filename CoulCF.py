import math,cmath

def cf1(x,eta,zl,eps,limit):
#
#
# ***    evaluate cf1  =  f   =  f'(zl,eta,x)/f(zl,eta,x)
#
#        using real arithmetic
    fpmin = 1e-50
    fpmax = 1./fpmin
    small    = math.sqrt(fpmin)
#
    fcl = 1.0
    xi = 1.0/x
    pk  = zl + 1.0
    px  = pk  + limit
    ek  = eta/pk
    f   =  ek + pk*xi
    if abs(f)<fpmin: f = fpmin
    d = 0.0
    c = f
    rk2 = 1.0 + ek*ek
    etane0 = eta != 0.0
    df = 0.
    #
# ***   begin cf1 loop on pk = k = lambda + 1
#
    while pk < px and abs(df-1.) > eps:
        pk1 = pk + 1.0
        tpk1 = pk + pk1
        if etane0:
            ek  = eta / pk
            rk2 = 1.0 + ek*ek
            tk  = tpk1*(xi + ek/pk1)
        else:
            tk  = tpk1*xi

        c  =  tk - rk2 / c
        d  =  tk - rk2 * d
        if abs(c)<fpmin: c = fpmin
        if abs(d)<fpmin: d = fpmin
        d = 1.0/d
        df = d * c
        f  = f * df
        fcl = fcl * d * tpk1*xi
        if abs(fcl)<small: fcl = fcl / small
        if abs(fcl)>fpmax: fcl = fcl * fpmin
        pk = pk1
        if pk > px: break

    nfp = pk - zl - 1
    if pk>px:
        print('cf1 has failed to converge after ',limit,' iterations as x =',x)
        err = 2.0
    else:
        err = eps * math.sqrt(nfp)
#     return(f,nfp,err)
    return(f)




def cf2(x,eta,zl,pm,eps,limit,acc8):
#
#                                    (omega)        (omega)
# *** Evaluate  CF2  = p + PM.q  =  H   (ETA,X)' / H   (ETA,X)
#                                    ZL             ZL
#     where PM = omega.i
#
    err = 1.0
    cf2 = 0.0
    ta = 2*limit
    e2mm1 = eta*eta + zl*zl + zl
    etap = eta * pm
    xi = 1./x
    wi = 2.*etap
    rk = 0.
    pq = (1. - eta*xi) * pm
    aa = -e2mm1 + etap
    bb = 2.*(x - eta + pm)
    rl = xi * pm
    if abs(bb) < eps: 
        if abs(aa + rk + wi) < eps: return(pq,eps)
        rl = rl * aa / (aa + rk + wi)
        pq = pq + rl * (bb + 2.*pm)
        aa = aa + 2.*(rk+wi)
        bb = bb + 4.*pm
        rk = rk + 4.

    if abs(bb) < eps: return (pq,abs(bb))
    dd = 1./bb
    dl = aa*dd* rl
    err = 1.
    while err > max(eps,acc8*rk*0.5)  and  rk < ta:
        pq = pq + dl
        rk = rk + 2.
        aa = aa + rk + wi
        bb = bb + 2.*pm
        dd = 1./(aa*dd + bb)
        dl = dl*(bb*dd - 1.)
        err = abs(dl)/abs(pq)

    pq = pq + dl
    if rk//2 > limit-1: 
        print('cf2(%i) not converged fully in %i iterations, so error in irregular solution = %10.2s  at zl = (%8.3f,%8.3f)' % (int(pm.imag),rk//2,err,zl.real,zl.imag))
    return (pq, err)


def csigma(Lmax,eta):
    import cmath,numpy,math
    from scipy.special import loggamma
    csig = numpy.zeros(Lmax+1)
    s = loggamma(complex(1,eta.real))
    csig[0] = s.imag
    for L in range(1,Lmax+1):
        csig[L] = csig[L-1] + math.atan2(eta.real,float(L))
    return(csig)

def dlde_steed(rho,eta,l,zi, acc,max_iter,acc8):

    aa=-eta*eta-float(l*(l+1))+zi*eta
    aap=eta*(eta-0.5*zi)/rho**2
    bb=2.0*(rho-eta+zi)
    bbp=(1.0+eta/rho)/rho
    dd=1.0/bb
    ddp=-dd**2*bbp
    dh=zi*aa*dd
    dhp=zi*(aap*dd+aa*ddp)
    hh=zi*(rho-eta)+dh
    hhp=zi*(1.0+eta/rho)/(2.0*rho)+dhp
    n=1
    nzl=0
    ndzl=0
    err1=abs(dh/hh)
    err2=abs(dhp/hhp)
#     print('dlde_steed:',err1,err2,acc,acc8*n,'so',max(err1,err2) > max(acc,acc8*n) , n < max_iter)
    while max(err1,err2) > max(acc,acc8*n) and n < max_iter:
#         print('n,errs:',n,err1,err2)
        n += 1
        aa=aa+float(2*n-2)+2.0*zi*eta
        aap=aap-zi*eta/rho**2
        bb=bb+2.0*zi
        tmp=1.0/(dd*aa+bb)
        ddp=-(ddp*aa+dd*aap+bbp)*tmp**2
        dd=tmp
        tmp=bb*dd-1.0
        dhp=(bbp*dd+bb*ddp)*dh+tmp*dhp
        dh=tmp*dh
        hh=hh+dh
        hhp=hhp+dhp
        
        err1=abs(dh/hh)
        err2=abs(dhp/hhp)
        
#     if n > max_iter: print('Max iterations exceeded in dlde_steed')
#     print(n,'dlde_steed:',err1,err2,acc,acc8*n,'so',max(err1,err2) > max(acc,acc8*n) , n < max_iter, 'for rho=',rho)
    return(hh,hhp)

    
def Pole_Shifts(S_poles,dSdE_poles, E_poles,has_widths, seg_val,lab2cm,QI,fmscal,rmass,prmax, etacns,za,zb,L_val):  # return new values in S_poles and dSdE_poles
    n_jsets,n_poles = E_poles.shape
    n_chans = seg_val.shape[1]
    for jset in range(n_jsets):
        for n in range(n_poles):
            if has_widths[jset,n]==0: continue
            for c in range(n_chans):
                pair = seg_val[jset,c] 
                if pair < 0: continue
                E = E_poles[jset,n]*lab2cm + QI[pair]
                if abs(E) < 1e-3: E = 1e-3
                
                sqE = cmath.sqrt(E)
                c_E = prmax[pair] * math.sqrt(fmscal*rmass[pair]) 
                c_eta = etacns * za[pair]*zb[pair] * cmath.sqrt(rmass[pair])
                
                rho = c_E*sqE
                eta = c_eta/sqE
#                 if E < 0: eta = -eta  #  negative imaginary part for bound states
                
                if abs(rho) <1e-10: 
                    S_poles[jset,n,c] = 0.0
                    continue
                EPS=1e-10; LIMIT = 2000000; ACC8 = 1e-12; L = 0; PM = complex(0.,1.)

                L = L_val[jset,c]
                zL,zLp = dlde_steed(rho,eta,L,PM, EPS,LIMIT,ACC8)
                S_poles[jset,n,c] = zL.real
                dSdE_poles[jset,n,c] = zLp.real * lab2cm * c_E**2

    return()
