
lightnuclei = {'n':'n', 'H1':'p', 'H2':'d', 'H3':'t', 'He3':'h', 'He4':'a', 'photon':'g'}

def printExcitationFunctions(XSp_tot_n,XSp_cap_n,XSp_mat_n, pname,tname, za,zb, npairs, base,n_data,E_scat,cm2lab,QI,ipair):
        
    for pin in range(npairs):

        pn = lightnuclei.get(pname[pin],pname[pin])
        tn = lightnuclei.get(tname[pin],tname[pin])
        neut = za[pin]*zb[pin] == 0    # calculate total cross-sections for neutrons
        fname = base + '-ftot_%s' % pn
        cname = base + '-fcap_%s' % pn
            
        print('Total cross-sections for incoming',pin,'to file',fname,' and capture to',cname)
        fout = open(fname,'w')
        cout = open(cname,'w')
        for ie in range(n_data):
#           E_scat[ie]      is lab incident energy in nominal entrance partition  ipair
#                 E = E_scat[ie]      # lab incident energy
#                 E = Ein_list[ie]    # incident energy in EXFOR experiment
            E = E_scat[ie]/cm2lab[ipair] + QI[pin] - QI[ipair]
            Elab = E * cm2lab[pin]
        
            x = XSp_tot_n[ie,pin] 
            print(Elab,x, file=fout)
            c = XSp_cap_n[ie,pin] 
            print(Elab,c, file=cout)
        fout.close()
        cout.close()

        for pout in range(npairs):
            if pin==pout and not neut: continue
            po = lightnuclei.get(pname[pout],pname[pout])
            to = lightnuclei.get(tname[pout],tname[pout])
            fname = base + '-fch_%s-to-%s' % (pn,po)
            print('Partition',pin,'to',pout,': angle-integrated cross-sections to file',fname)
            fout = open(fname,'w')
#                     fouo = open(fname+'@','w')
        
            for ie in range(n_data):
                x = XSp_mat_n[ie,pout,pin]
#                     E = E_scat[ie]
#                     E = Ein_list[ie]
                E = E_scat[ie]/cm2lab[ipair] + QI[pin] - QI[ipair]
                Elab = E * cm2lab[pin]
                print(Elab,x, file=fout)
#                         print(Ein_list[ie],x, Elab,pin,ipair,1./cm2lab[ipair],cm2lab[pin],cm2lab[pin]/cm2lab[ipair],ie,'/',file=fouo)
            fout.close()
    return
