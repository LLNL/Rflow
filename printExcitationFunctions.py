
##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

import numpy
lightnuclei = {'n':'n', 'H1':'p', 'H2':'d', 'H3':'t', 'He3':'h', 'He4':'a', 'photon':'g'}
from PoPs.chemicalElements.misc import *

def nuclIDs (nucl):
    datas = chemicalElementALevelIDsAndAnti(nucl)
    if datas[1] is not None:
        return datas[1]+str(datas[2]),datas[3]
    else:
        return datas[0],0

def quickName(p,t):     #   (He4,Be11_e3) -> a3
    ln = lightnuclei.get(p,p)
    tnucl,tlevel = nuclIDs(t)
    return(ln + str(tlevel) if tlevel>0 else ln)
        
# based on printExcitationFunctions4 

def printExcitationFunctions(XSp_tot_n,XSp_cap_n,XSp_mat_n, pname,tname, za,zb, npairs, base,n_data,E_scat,EIndex,cm2lab,QI,ipair,Eframe):
        
#  Eframe False:   plot as function of cm energy in the partition ipair (the projectile frame in gnds source)
#  Eframe True:    plot as function of lab energy in the incident partition for each transitiion.
#  Name files *-g* and *-f* respectively
    sym = "f" if Eframe else "G"
    
    if not Eframe:
        totals = numpy.zeros([npairs+1,npairs,n_data])
    else:
        totals = None
#     Gnames = []
    pnin = None
    for pin in range(npairs):

        pn = quickName(pname[pin],tname[pin])
        neut = za[pin]*zb[pin] == 0    # calculate total cross-sections for neutrons
        if neut: pnin = pin
        fname = base + '-%stot_%s' % (sym,pn)
        cname = base + '-%scap_%s' % (sym,pn)
        fname_e = fname if '/' not in fname else fname.split('/')[1]
        cname_e = cname if '/' not in cname else cname.split('/')[1]
#         Gnames.append(fname)
            
        print('    Total cross-sections for incoming',pin,'to file',fname_e,'\n         and capture to',cname_e)
        fout = open(fname,'w')
        cout = open(cname,'w')
        for ie0 in range(n_data):
            ie = EIndex[ie0]
#           E_scat[ie]      is lab incident energy in nominal entrance partition  ipair
#                 E = E_scat[ie]      # lab incident energy
#                 E = Ein_list[ie]    # incident energy in EXFOR experiment

            x = XSp_tot_n[ie,pin] 
            c = XSp_cap_n[ie,pin] 
            if Eframe:
                E = E_scat[ie]/cm2lab[ipair] + QI[pin] - QI[ipair]
                Elab = E * cm2lab[pin]
                Eout = Elab
            else:
                Eout = E_scat[ie]
                totals[-1,pin,ie] = x
        
            if x>0: print(Eout,x, file=fout)
            if c>0: print(Eout,c, file=cout)
        fout.close()
        cout.close()

        for pout in range(npairs):
            if pin==pout and not neut: continue
            po = quickName(pname[pout],tname[pout])
            fname = base + '-%sch_%s-to-%s' % (sym,pn,po)
            fname_e = fname if '/' not in fname else fname.split('/')[1]
            print('        Partition',pin,'to',pout,': angle-integrated cross-sections to file',fname_e)
            fout = open(fname,'w')
#             Gnames.append(fname)
#                     fouo = open(fname+'@','w')
        
            for ie0 in range(n_data):
                ie = EIndex[ie0]
                x = XSp_mat_n[ie,pout,pin]
#                     E = E_scat[ie]
#                     E = Ein_list[ie]
                if Eframe:
                    E = E_scat[ie]/cm2lab[ipair] + QI[pin] - QI[ipair]
                    Elab = E * cm2lab[pin]
                    Eout = Elab
                else:
                    Eout = E_scat[ie]
                    totals[pout,pin,ie] = x
                if x>0: print(Eout,x, file=fout)
#                         print(Ein_list[ie],x, Elab,pin,ipair,1./cm2lab[ipair],cm2lab[pin],cm2lab[pin]/cm2lab[ipair],ie,'/',file=fouo)
            fout.close()
            
#     print('Plot global integrated data:\n xmgr  \\')
#     for i in range(len(Gnames)):
#         print(' -graph %s  %s \\' % (i,Gnames[i]) )
#     print('  -rows %s \n' % len(Gnames) )
    
    return(pnin,totals)
