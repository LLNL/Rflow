#! /usr/bin/env python3

##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################


import os,sys,copy,csv,math
import argparse
from CoulCF import dSoP
from PoPs import database as databaseModule
from nuclear import *
pi = math.pi
rad = pi/180

hbc =   197.3269788e0             # hcross * c (MeV.fm)
finec = 137.035999139e0           # 1/alpha (fine-structure constant)
amu   = 931.4940954e0             # 1 amu/c^2 in MeV

coulcn = hbc/finec                # e^2
fmscal = 2e0 * amu / hbc**2
etacns = coulcn * math.sqrt(fmscal) * 0.5e0

hbar = 6.582e-22 # MeV.s

EPS = 1e-6 # upper bound on accuracy of Coulomb functions from CoulCF.py

print('\nlevels2sfresco 0.20')
cmd = ' '.join([t if '*' not in t else ("'%s'" % t) for t in sys.argv[:]])
print('  Command:',cmd ,'\n')

defaultPops = '../ripl2pops_to_Z8.xml'
lightnuclei = {'n':'n', 'H1':'p', 'H2':'d', 'H3':'t', 'He3':'h', 'He4':'a', 'photon':'g'}

fresco_start = \
"""R-matrix starter for {%s}
NAMELIST
&FRESCO
 hcm=0.1000 rmatch=%.4f rintp=0.5000
 jtmin=0.0 jtmax=%s absend=-1.0
 thmin=30 thmax=-180 thinc=50  smats=0 weak=0
 iter = 0 nrbases=1 nparameters=0 btype="A"  boldplot=F
 pel=1 exl=1 elab(1:2) = %s %s nlab(1)=1 /
"""
 
def make_fresco_input(projs,targs,masses,charges,qvalue,levels,pops,Jmax,Projectiles,EminCN,EmaxCN,emin,emax,
        Rmatrix_radius,jdef,pidef,widef,Term0,gammas,ReichMoore,outbase,MaxPars,FormalWidths):

#     pops = databaseModule.Database.read(popsicles[0])
#     for p in popsicles[1:]:
#         print('read further pops file',p)
#         pops.addFile(p)
    gs = '+g' if (gammas or ReichMoore) else ''

    fresco = open('fresco%s.in' % gs,'w')
    pel = 1  # in the made fresco files. This sets zero of pole energies
    Qpel = None
    
    print(' CN partition is ',projs.index('photon'))

    mxp = 0
    reaction = ', '.join([("%s+%s" % (projs[ic],targs[ic]) if projs[ic]!='photon' or gammas else '') for ic in range(len(projs)) ]) 
    CN = None
    OtherSep = -1e6
    excitationLists = []
    partitions = []
    Lvals = None
    for ic in range(len(projs)):
        p = projs[ic]
        t = targs[ic]
        pz = charges[p]
        tz = charges[t]
        pmass = masses[p];  Ap = int(pmass+0.5)
        tmass = masses[t];  At = int(tmass+0.5)
        prmax = Rmatrix_radius * (Ap**(1./3.) + At**(1./3.)) if Rmatrix_radius > 0 else abs(Rmatrix_radius)
        Q = qvalue[p]

        if ic==0: # put in namelist start
            starting = fresco_start % (reaction,prmax,Jmax,emin,emax)
            print(starting, file=fresco)
        
        if p == 'photon': 
            CN = t
            CNsep = Q
            if not gammas: continue
        else:
            OtherSep = max(OtherSep,Q)
        if Qpel is None: Qpel = qvalue['photon'] - qvalue[p]  # first partition not excluded

        if p not in Projectiles: continue    
        if emax + Q < 1.:
            print('Omitting partition %s+%s as not open even at %s MeV' % (p,t,emax))
            continue

        nex = 1
        jp = (Ap % 2) * 0.5 if Ap!=2 else 1.0
        pp = 1  # all stable projectiles  A<=4 are + parity
        ep = 0.0

#         print('For target:',t)
        tgs,tlevel = t.split('_e') if '_e' in t else t,0
        if t=='N0': t='N14'
        target = pops[t.lower()]
        proj   = pops[p.lower()]

        jt,pt,et = target.spin[0].float('hbar'), target.parity[0].value, target.energy[0].pqu('MeV').value
#         jp,pp,ep =   proj.spin[0].float('hbar'),   proj.parity[0].value,   proj.energy[0].pqu('MeV').value
    

        levelList = levels[t]
        nex = len(levelList)
        print("\n &PARTITION namep='%s' massp=%s zp=%s nex=%s namet='%s' masst=%s zt=%s qval=%s prmax=%.4f/" % (p,pmass,pz,nex,t,tmass,tz,Q,prmax), file=fresco)
        partitions.append(ic)
        mxp += 1
        excitationPairs = []
        print("  &STATES  cpot =%s jp=%s ptyp=%s ep=%s  jt=%s ptyt=%s et=%s /" % (mxp,jp,pp,ep,jt,pt,et) , file=fresco)
        excitationPairs.append([jp,pp,ep,jt,pt,et])
        for level in levelList:
            if level == 0: continue
            targetex = pops["%s_e%s" % (tgs,level)].nucleus
            print('R*',targetex.id,targetex.spin[0],targetex.parity[0],targetex.energy[0])
            jt,pt,et = targetex.spin[0].float('hbar'), targetex.parity[0].value, targetex.energy[0].pqu('MeV').value
            print("  &STATES  copyp=1                       jt=%s ptyt=%s et=%s /" % (jt,pt,et) , file=fresco)
            excitationPairs.append([jp,pp,ep,jt,pt,et])
        excitationLists.append(excitationPairs)
        
    print(" &PARTITION /\n", file=fresco)
    
    for ic in range(mxp):
        print(" &pot kp=%s  type=0 p(1:3) = 0 0 1.0 /" % (ic+1), file=fresco)
    print(" &pot /\n\n &overlap /\n\n &coupling /\nEOF\n", file=fresco)
    
    if CN is None:
        print('CN not determined for resonances')
        return
        
    CNsep = CNsep - OtherSep
    print('Include CN resonances from threshold at %7.3f MeV excitation, up to %7.3f MeV' % (CNsep,EmaxCN))
    print('Zero pole energy in fresco is at %7.3f excitation' % Qpel,'\n')
    print('Remaining partitions:',partitions)
    print('excitationLists:',excitationLists)
    
    cn = CN.lower()
    CNresonances = []
    for nucl in pops.keys():
        if nucl.startswith(cn):
            CNresonances.append(pops[nucl])
#             print('Include',nucl,'resonance')
    if len(CNresonances)==0: return
    
    frescoVars = open('fresco%s.vars' % gs,'w')
    term = Term0
    nvars = 0
    spinGroups = {}
    for level in CNresonances:
        print()
        jt,Er = level.spin[0].float('hbar'), level.energy[0].pqu('MeV').value
        if jt<0:
            print('Level at %s MeV of unknown spin and parity' % Er, "Set to",jdef)
            jt = jdef
            
        try:
            pt = level.parity[0].value
        except:
            print('Level at %s MeV of unknown parity' % Er, "Set to",pidef)
            pt = pidef

        if Er < EminCN or Er > EmaxCN: continue
        
        try:
            halflife = level.halflife[0].pqu('s').value
            width = hbar/halflife*math.log(2)
        except:
            width = None 
        try:
            print('Include',level.id,'resonance at',Er,'MeV','halflife:',level.halflife[0].pqu('s').value if level.halflife is not None else None, 'Width:',width,'MeV')
        except:
            print('Include',level.id,'resonance at',Er,'MeV')

        step = 0.1 # search step in MeV
        parity = '+' if pt > 0 else '-'
        name = 'J%s%s:E%s' % (jt,parity,Er)
        term += 1

            
        Epole = Er - Qpel   # fresco convention for R-matrix poles
        print("\n&Variable kind=3 name='%s' term=%s jtot=%s par=%s energy=%s step=%s / obs width ~ %6s" % (name,term,jt,pt,Epole,step,width), file=frescoVars)
        print('Epole:',Epole,'(cm). %.1f%s' % (jt,parity), ':',name)
        nvars += 1
        
        if ReichMoore:
            damping = 3.3e-7 # MeV
            dname = 'D:' + name
            print("&Variable kind=7 name='%s' term=%s damping=%s  step=%s / obs width ~ %6s" % (dname,term,damping,step,width), file=frescoVars)
            nvars += 1

        JJ = jt
        pi = pt
        weight = 0
        channels = []
        icnew = -1
        for ic in partitions:  # retained partitions only
            p = projs[ic]
            t = targs[ic]
            pz = charges[p]
            tz = charges[t]
            pmass = masses[p];  Ap = int(pmass+0.5)
            tmass = masses[t];  At = int(tmass+0.5)
            prmax = Rmatrix_radius * (Ap**(1./3.) + At**(1./3.)) if Rmatrix_radius > 0 else abs(Rmatrix_radius)
            rmass = pmass*tmass/(pmass+tmass)
            Qp = qvalue[p]
            if rmass > 0.:
                wignerLimitRWA = ( 3.0/prmax**2  / (fmscal * rmass) ) ** 0.5
            else:
                wignerLimitRWA = 0.1

            print('\n##\nPartition',ic,':',p,t,'so Q=',Qp,'  Wigner rwa=',wignerLimitRWA)
            icnew += 1
             
            ia = 0
            excitationList = excitationLists[icnew]
            for excitationPair in excitationList:
                ia += 1
                jp,pp,ep,jt,pt,et = excitationPair
                Q = Qp - et
                smin = abs(jt-jp)
                smax = jt+jp
                s2min = int(2*smin+0.5)
                s2max = int(2*smax+0.5)
                for s2 in range(s2min,s2max+1,2):
                    sch = s2*0.5
                    lmin = int(abs(sch-JJ) +0.5)
                    lmax = int(sch+JJ +0.5)
                    if Lvals is not None: lmax = min(lmax,lMax[icch-1])
                    first = 1
                    for lch in range(lmin,lmax+1):
                        if pi != pp*pt*(-1)**lch: continue
                        if Epole < 0 and lch > abs(pz) : continue                   # only allow s-wave neutrons, s,p-wave protons, etc in closed channels
                        # print(' Partial wave channels IC,IA,L,S:',ic,ia,lch,sch)
                        w = 1./(2*lch+1)**2
                        widthCH = widef
                        print('--')
                        if Epole+Q > 0. and rmass>0:    # don't bother for sub-threshold states 
                            dSoPc,P = dSoP(Epole, Q,fmscal,rmass,prmax, etacns,pz,tz,lch)
#                             print('dSoP,P(',Epole, Q,fmscal,rmass,prmax, etacns,pz,tz,lch, ') = ',dSoPc,P)
#                             print('w,width,dSoPc:',w,width,dSoPc)
                            if abs(P) < EPS: 
                                widthCH = 0.0
                                channels.append((icnew+1,ia,lch,sch,ic,et,widthCH))
                                continue

                            if dSoPc is None: dSoPc = 0.
                            WignerFW = 2*wignerLimitRWA**2*P
                            if width is None:
                                widthCH = WignerFW * widef   # starting point for fitting
                            else:
                                widthCH = width
                            if abs(widthCH) > WignerFW:
                                widthCH = WignerFW if widthCH > 0 else -WignerFW                            
                                
                            if not FormalWidths: 
                                w *= ( 1 - widthCH * dSoPc/2.)
                            w *= first   # more weight on lowest L !!!!
                            rwa = abs(widthCH/(2*P.real))**0.5
                            print('For ch',(ic,ia,lch,sch,widthCH),'Er,Epole,Q,P =%7.3f, %7.3f, %7.3f, %.3e' %(Er, Epole,Q,P), 'WigFW=%6.2f, rwa=%6.2f' % (WignerFW,rwa) )#'dSoPc=',dSoPc, width*dSoPc/2.,'w =',w)
                            weight += w
                        elif rmass == 0.0:  # photons
                            w = 1e-4
                            weight += w
                            print('For photon ch',(ic,ia,lch,sch),'Er,Epole,Q =%7.3f, %7.3f, %7.3f,' %(Er, Epole,Q),'w =',w)
                        else:
                            dSoPc = 0.
                            print('Closed ch',(ic,ia,lch,sch),'Er,Epole,Q =%7.3f, %7.3f, %7.3f,' %(Er, Epole,Q) ) #  ,'dSoPc=',dSoPc, width*dSoPc/2.,'w =',w)
                            widthCH = 0.0
                        channels.append((icnew+1,ia,lch,sch,ic,et,widthCH))
                             
                        
        nChans = len(channels)
        Jpi = '%s,%s' % (JJ,pi)
        spinGroups[Jpi] = channels
        c = 0
        print()
        for icch,iach,lch,sch,ic,et,widthCH in channels:
            Ec = Epole+qvalue[projs[ic]] - et
            print('Ch:',icch,iach,lch,sch,ic,'with p,Q,Ec,widthCH =',projs[ic],qvalue[projs[ic]],Ec,widthCH)
            w = 1./(2*lch+1)**2  # if Ec > 0. else 0  # channel not open: does not contribute to widths
            wRel = w/(weight+1e-10) if weight > 0 else w
            c += 1
            stepFactor = 1e-2
            pWid = widthCH*wRel
            print("&Variable kind=4 name='w%s,%s' term=%s icch=%s iach=%s lch=%s sch=%s width=%s rwa=F step= %9.2e/ for E=%s" % (c,name,term,icch,iach,lch,sch,pWid,pWid*stepFactor,Ec), file=frescoVars)
            nvars += 1
            
    resonanceTerms = term
    
    BackGroundTerms = False
    if BackGroundTerms: 
        EBG = 40.0
        wBG = 1
        step = 0.1
        for spinGroup in spinGroups.keys():
            channels = spinGroups[spinGroup]
            JJ,pi = spinGroup.split(',')
            JJ,pi = float(JJ),int(pi)
            Epole = EBG - Qpel   # fresco convention for R-matrix poles
            term += 1
            parity = '+' if pi > 0 else '-'
            name = 'BG:J%s%s' % (JJ,parity)
            print("\n&Variable kind=3 name='%s' term=%s jtot=%s par=%s energy=%s step=%s / obs width ~ %6s" % (name,term,JJ,pi,Epole,step,wBG), file=frescoVars)
            nvars += 1 
            c = 0
            for icch,iach,lch,sch,ic,et,widthCH  in channels:
                c += 1
                print("&Variable kind=4 name='w%s,%s' term=%s icch=%s iach=%s lch=%s sch=%s width=%s rwa=F step= %9.2e/ for E=%s" % (c,name,term,icch,iach,lch,sch,wBG,wBG*step,EBG), file=frescoVars)
                nvars += 1     
        
        
        
    print('%s resonance levels' % resonanceTerms,', with BG:',term,'\n')
    fresco.close()
    frescoVars.close()
   
#     sfrescoName = "%sr%s.sfresco" % (CN,gs)
    pel = pel-1
    print(projs)
    pq = lightnuclei.get(projs[pel],projs[pel])
    sfrescoName = "%s%s%s-%s.sfresco" % (CN,pq,gs,outbase)
    sf = open(sfrescoName,'w')
    fLines = open('fresco%s.in' % gs,'r').readlines()
    vLines = open('fresco%s.vars' % gs,'r').readlines()
    print(" '='  '%s.frout' " % (sfrescoName+gs), file=sf )
    if MaxPars is None:
        print(nvars,0,'\n', file=sf) 
    else:
        print(min(nvars,MaxPars),0,'\n', file=sf) 
    sf.writelines(fLines)
    if MaxPars is None:
        sf.writelines(vLines)
    
    else:
        sf.writelines(vLines[:MaxPars+1])

    return
    


# Process command line options
parser = argparse.ArgumentParser(description='Prepare data for Rflow')
parser.add_argument('CN', type=str, help='Compound nucleus' )

parser.add_argument("-P", "--Projectiles", type=str, nargs='+', help="List of projectiles (gnds names). First is projectile in GNDS file")
parser.add_argument("-L", "--LevelsMax", type=int, nargs='+', help="List of max level numbers of corresponding targets, in same order as -P")
parser.add_argument("-B", "--EminCN", type=float, help="Minimum energy relative to gs of the compound nucleus.")
parser.add_argument("-C", "--EmaxCN", type=float,  help="Maximum energy relative to gs of the compound nucleus.")
parser.add_argument("-J", "--Jmax", type=float, default=8.0, help="Maximum total J of partial wave set.")
parser.add_argument("-e", "--eminp", type=float, default=0.1, help="Minimum incident lab energy in first partition.")
parser.add_argument("-E", "--emaxp", type=float, default=25., help="Maximum incident lab energy in first partition.")
parser.add_argument("-r", "--Rmatrix_radius", type=float, default=-10, help="Reduced R-matrix radius: factor of (A1^1/3+A2^1/3).")
parser.add_argument("-G", "--GammaChannel", action="store_true", help="Include discrete gamma channel")
parser.add_argument("-R", "--ReichMoore", action="store_true", help="Inclusive capture channel")
parser.add_argument("-j", "--jdef", type=float, default=2.0, help="Default spins for unknown RIPL states")
parser.add_argument("-p", "--pidef", type=int, default=1, help="Default spins for unknown RIPL states")
parser.add_argument("-w", "--widef", type=float, default=0.10, help="Default width (MeV) for unknown RIPL states")
parser.add_argument("-F", "--FormalWidths", action="store_true", help="Treat widths as already formal widths.")

parser.add_argument(      "--pops", type=str, default=defaultPops, help="pops files with all the level information from RIPL3. Default = %s" % defaultPops)
parser.add_argument(      "--pops2", type=str, help="local pops file")


parser.add_argument("-T", "--Term0", type=int, default=0, help="First 'term' in Sfresco output")
parser.add_argument("-M", "--MaxPars", type=int, help="Max numver of R-matrix parameters")


# print('Command:',' '.join(sys.argv[:]) ,'\n')
    
args = parser.parse_args()

# os.system('mkdir '+Dir)
EmaxCN = args.EmaxCN
Projectiles = args.Projectiles
LevelsMax = args.LevelsMax
pops = databaseModule.Database.read( args.pops )
if args.pops2 is not None: pops.addFile( args.pops2 , replace=True)
    
scales = {-1: "nodim", 0: "fm^2", 1: "b", 2:"mb", 3:"mic-b"}
rscales = {"nodim": -1, "fm^2":0, "b":1, "mb":2, "mic-b":3, "microbarns":3}
xscales = {-1:1,  0:10, 1:1000, 2:1, 3:1e-3}
lightA      = {'p':'H1', 'd':'H2', 't':'H3', 'h':'He3', 'a':'He4', 'g':'photon'}

amu   = 931.4940954e0             # 1 amu/c^2 in MeV

props = {}

projs = [];    targs = []; levels = []
masses={}; qvalue = {}; charges={}

CN = args.CN
CNp = pops[CN]
cnMass = CNp.getMass('amu')
if hasattr(CNp, 'nucleus'): CNp = CNp.nucleus
cnCharge = CNp.charge[0].value
cnA = CNp.A

MP = len(args.Projectiles)
projs = args.Projectiles 
print('projs 1',projs)
if 'photon' not in projs: 
    projs += ['photon']
masses['photon'] = 0.0
charges['photon'] = 0.0
print('projs 2',projs)
MPT = len(projs)
targs = []
elimits = [0 for i in range(MPT)]
levels = {}

# print('args.LevelsMax',args.LevelsMax)
ip = 0
name = 'P'
for proj in projs:
    if proj in ['TOT','.']: continue
    n = proj if proj != '2n' else 'n'
    try:
        pe = pops[n]
    except:
        print('Nuclide',n,'not in database!!  SKIP')
        continue 
    masses[proj] = pe.getMass('amu')
    if hasattr(pe, 'nucleus'): pe = pe.nucleus
    charges[proj] = pe.charge[0].value

    Ztarg = cnCharge - charges[proj]
    Mtarg = cnMass - masses[proj]
    targ = elementsZSymbolName[Ztarg][0] 
    Atarg = int(0.5 + Mtarg)
    targ = targ+str(Atarg)
    targs += [targ]
    t = pops[targ]
    if hasattr(t, 'nucleus'): t = t.nucleus
    masses[targ] = t.getMass('amu')
    charges[targ] = Ztarg
    if ip < MP:
        levs = set()
        for level in range(args.LevelsMax[ip]+1):
            levs.add(level)
        levels[targ] = levs
        name += lightnuclei[proj]+str(args.LevelsMax[ip])
    ip += 1

# print('targs',targs)


p_ref = args.Projectiles[0]
t_ref = targs[projs.index(p_ref)]
# print('masses:\n',masses)
masses_ref = masses[p_ref] + masses[t_ref]

try:
    ipi = args.Projectiles.index('photon')
    ZT = int(charges[p_ref]+charges[t_ref])
    AT = int(0.5+ masses[p_ref]+masses[t_ref])
    targs[ipi] = '%s%s' % (elementSymbolFromZ(ZT),AT)
    print('From Z,A=',ZT,AT,'have target',targs[MP] )
    masses[targs[ipi]] = pops[targs[ipi]].getMass('amu')
    charges[targs[ipi]] = ZT
except:
    pass

print('Partitions with projs:',projs,' targs:',targs)
print('           masses: ',masses)
for ipi in range(MPT):
    p = projs[ipi]
    t = targs[ipi]
#     print('partition',ipi,' p,t=',p,t)
    masses_i = masses[p] + masses[t]
    qvalue[p] = (masses_ref - masses_i) * amu
    elimits[ipi] = args.EmaxCN 

print('Partitions with projs:',projs,' targs:',targs)
print('Partitions elimits:',elimits)
print('           masses: ',masses)
print('           charges: ',charges)
print('           Q values: ',qvalue)
if len(projs)==0:
    print("Missing information about projectiles for partitions and limits")
    sys.exit(1)


print("Excited states used:",levels)

print('\nSFresco input file:')
make_fresco_input(projs,targs,masses,charges,qvalue,levels,pops,args.Jmax,Projectiles,
    args.EminCN,args.EmaxCN,args.eminp,args.emaxp,args.Rmatrix_radius,
    args.jdef,args.pidef,args.widef,args.Term0,args.GammaChannel,args.ReichMoore,
    name,args.MaxPars,args.FormalWidths)

