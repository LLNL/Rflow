#! /usr/bin/env python

import os,sys,copy,csv,math
import argparse
pi = math.pi
rad = pi/180

hbc =   197.3269788e0             # hcross * c (MeV.fm)
finec = 137.035999139e0           # 1/alpha (fine-structure constant)
amu   = 931.4940954e0             # 1 amu/c^2 in MeV

coulcn = hbc/finec                # e^2
fmscal = 2e0 * amu / hbc**2
etacns = coulcn * math.sqrt(fmscal) * 0.5e0

def lab2cm(mu_lab, ap,at,ae,ar, E_lab,Q):
#
#  p + t -> e + r (projectile + target -> ejectile + residual
#
#  Inverse kinematics (when ap >> at) usually gives x > 1.
#  Then, only a limited range of sin_lab < 1/x allows th_cm to have any solution.
#        In that case, need to take both branches of the math.asin() function to get 2 solutions.
#
    E_cm = at/(ap+at) * E_lab
    E_out = E_cm + Q

    x = ((ap*ae * E_cm) / (at*ar * E_out))**0.5
    th_lab = math.acos(mu_lab)
    sin_lab = (1-mu_lab**2) ** 0.5
    th_cm = th_lab + math.asin(x*sin_lab)
    cos_cm = math.cos(th_cm)
    cos_cm = max(cos_cm,-1.0+1e-5)    # avoid cm=180 deg in case x=1, to avoid 0/0
    a = 1 + x*cos_cm
    b = abs(a)
    frame_scale = b/(1.+x*x+2.*x*cos_cm)**1.5   # sigCM=sigLAB*frame_scale or  sigLAB=sigCM/frame_scale
    return (cos_cm, frame_scale)
    
# Process command line options
parser = argparse.ArgumentParser(description='Prepare data for Rflow')

parser.add_argument("Projectile", type=str,  help="gnds name for projectile in GNDS file to be used. For all Elab energies.")
parser.add_argument("Emax", type=float,  help="Maximum energy relative to gs of the compound nucleus.")

parser.add_argument('inFiles', type=str, nargs="+", help='cross-section data files: E,angle,expt,err as desribed in property file.')
parser.add_argument("-d", "--Dir", type=str,  default="Data", help="output data directory for small filesdu")
parser.add_argument("-o", "--Out", type=str,  default="flow.data", help="output data file")
parser.add_argument("-n", "--Norms", type=str,  default="flow.norms", help="output normalization file")
parser.add_argument("-S", "--Sfresco", action="store_true", help="Outputs for Sfresdco")

parser.add_argument("-p", "--Props", type=str,  default="properties", help="property file")
parser.add_argument("-a", "--Adjusts", type=str,  default="adjusts", help="list of current norm and shift adjustments")
parser.add_argument("-f", "--Fits", type=str,  default="datafit.csv", help="list of current norm and shift adjustments")


args = parser.parse_args()
Dir = args.Dir + '/'

    
scales = {-1: "nodim", 0: "fm^2", 1: "b", 2:"mb", 3:"mic-b"}
rscales = {"nodim": -1, "fm^2":0, "b":1, "mb":2, "mic-b":3, "microbarns":3}
xscales = {-1:1,  0:10, 1:1000, 2:1, 3:1e-3}
lightA      = {'p':'H1', 'd':'H2', 't':'H3', 'h':'He3', 'a':'He4', 'g':'photon'}

amu   = 931.4940954e0             # 1 amu/c^2 in MeV

sf = open(Dir + 'sfresco.split','w')
plotd = open(Dir + 'sfresco.split.plotd','w')
output = open(args.Out,'w')
noutput = open(args.Norms,'w')
z_errorOut = open(Dir + 'Zero-errors.log','w')
z_errors = set()
nsf = 0

print("\nProperty dictionary from '%s'" % args.Props)
print("                           p e r  ang-int  %   %   expect  group  split  lab    abs   iscale   Aflip  Ein   rRuth Sfactor  Escale Eshift Ecalib split run")
print("           File                           sys stat norm    an/en  norms  a,xs   err   units                                 shifts directory")
csv_out = open(Dir + args.Props + '-o.csv','w')
headings = ['projectile','ejectile','residual','file','integrated','sys-error','stat-error','norm','group','splitnorms','lab','abserr','scale','filedir','Aflip','Ein','ratioRuth','Sfactor','eshift','ecalib','splitshifts']
print(','.join(headings), file=csv_out)

props = {}

# csv     #   SPREADSHEET !
csvf =  open(Dir + args.Props,'r')
projs = [];    targs = []; elimits = []; masses={}; qvalue = {}; charges={}
for prop in csv.DictReader(csvf):
    projectile = prop['projectile']
    ejectile = prop['ejectile']
    residual = prop.get('residual',0)
    datFile = prop['file']
    sys_percent = prop['sys-error']
    stat_percent = prop['stat-error']
    expect  = prop['norm']
    group  = prop['group']
    if datFile == 'LIMIT':
        print('LIMIT')
        target = ejectile
        projs.append(projectile)
        targs.append(ejectile)
        elimits.append(float(residual))
        masses[projectile] = float(sys_percent)
        masses[target] = float(stat_percent)  # target mass!
        qvalue[projectile] = float(prop['angle-integrated'])
        charges[projectile] = int(expect)
        charges[target] = int(group)  # target charge!
        continue
#     print(( prop['angle-integrated']))
    integrated = prop['angle-integrated'][0]=='T'        
    expect  = prop['norm']
    group  = prop['group']
    splitgroupnorms = prop['splitnorms'][0]=='T'
    lab = prop['lab'][0]=='T'
    abserr = prop['abserr'][0]=='T'
    scale = prop['scale']
    filedir =prop['filedir']
    Aflip  = prop.get('Aflip')[0]=='T'
    Ein  = prop['Ein']
    rRuth  = prop.get('ratioRuth')[0]=='T'
    Sfactor = prop.get('S_factor')[0]=='T'
    eshift  = float(prop.get('eshift',0.))
    ecalib  = float(prop.get('ecalib',0.))
    splitgroupshifts = prop.get('splitshifts')[0]=='T'
    iscale = rscales[scale]
    print("%26s %1s %1s %1s  %5s %5s %5s %5s %5s     %5s %5s %5s %4s=%-5s %5s %5s %5s %5s %5s  %5s  %s %s" % (datFile,projectile,ejectile,residual,integrated,sys_percent,stat_percent,expect,group,splitgroupnorms,lab,abserr,iscale,scale,Aflip,Ein,rRuth,Sfactor,eshift,ecalib,splitgroupshifts,filedir))
    props[datFile] = (projectile,ejectile,residual,integrated,float(sys_percent)/100.,float(stat_percent)/100.,expect,group,splitgroupnorms,lab,abserr,iscale,Aflip,Ein,rRuth,Sfactor,eshift,ecalib,splitgroupshifts,filedir)
    pdata      = [projectile,ejectile,residual,datFile,integrated,sys_percent,stat_percent,expect,group,splitgroupnorms,lab,abserr,scale,filedir,Aflip,Ein,rRuth,Sfactor,eshift,ecalib,splitgroupshifts]
    sdata      = [str(d) for d in pdata]
    print(','.join(sdata), file=csv_out)
    stat_percent = float(stat_percent)

print('Partitions with projs:',projs,' targs:',targs,' elimits:',elimits)
print('           masses: ',masses)
print('           charges: ',charges)
if len(projs)==0:
    print("Missing information about projectiles for partitions and limits")
    sys.exit(1)
norm_adj = {}
shift_adj = {}

try:
    adjustments = f90nml.read(args.Adjusts)
    for dataset in adjustments['datasetnorm']:
        datanorm = dataset['datanorm']
        reffile  = dataset['reffile'].strip()
        norm_adj[reffile] = datanorm
    for dataset in adjustments['datasetshift']:
        dataeshift = dataset['dataeshift']
        reffile  = dataset['reffile'].strip()
        shift_adj[reffile] = dataeshift
    
    print("Norm adjustments:",len(list(norm_adj.keys())))
    print("Shift adjustments:",len(list(shift_adj.keys())))
    print("Norm adjustments:\n",norm_adj)
    # print "Shift adjustments:\n",shift_adj
except:
    print('No previous norms or energy shifts')

pargrp=0
parsplit = 0
grouplists = []
splitdata = [] 
datanorms = []
flownorms = []
datashifts = []
normlimits = []
shiftlimits = []
idlast = 0
idfirst = 1
ffirst = 1
flast  = 0
for datFile in args.inFiles:
    if '@' in datFile: continue
    baseFile = datFile.split(Dir)[1]
    d = open(datFile,'r')
    dr = datFile.split('.dat')[0]
    projectile,ejectile,residual,integrated,syserror,staterror,expect,group,splitgroupnorms,lab,abserr,iscale,Aflip,Ein,rRuth,Sfactor,eshift,ecalib,splitgroupshifts,filedir = props[baseFile]
    print("\nRead ",datFile," write root:",filedir + dr,"   A,E-flip=",Aflip,Ein,', s/R:',rRuth)

#   p,e,x = datFile.split('_')[1][0:3]
    p,e,x = projectile,ejectile,residual
    if p not in projs or (e not in projs and e != 'TOT'):
        print('SKIP',datFile,'as strange projectile')
        continue
    try:
        ia= int(x)+1
    except:
        ia = 1
    pel = projs.index(p) + 1
    ic  = projs.index(e) + 1 if e != 'TOT' else 0
    elim = elimits[pel-1]
    index = projs.index(p)
    t = targs[index]
    r = targs[projs.index(e)] if e!= 'TOT' else 0
    
    p_ref = args.Projectile
    t_ref = targs[projs.index(p_ref)]
    
    Ein_scale =  1.0
    Ec = Ein[0].lower()
    if Ec in ['c','p','t']:
        if Ec=='c':
            Ein_scale = (masses[t] + masses[p])/masses[t]
        elif Ec=='p':
            Ein_scale =  1.0
        elif Ec=='t':
            Ein_scale = masses[p]/masses[t]
        print("  Scale projectile energy by %.5f" % Ein_scale)
    idir = 1 if rRuth else 0
    pn,tn,en,rn = [lightA.get(n,n) for n in (p,t,e,r)]
    if e != 'TOT':
        Qvalue_masses = (masses[p] + masses[t] - masses[e]-masses[r]) * amu
        Qvalue = qvalue[e] - qvalue[p]
        print("Q value =",Qvalue,Qvalue_masses,' Target for gnds projectile =',args.Projectile)
    else:
        Qvalue = 0.
    lab2cm_in = masses[t]/(masses[p]+masses[t])
    
    Qvalue_ref = qvalue[p_ref] - qvalue[p]; print(' From p=',p,qvalue[p],' and p_ref',p_ref,qvalue[p_ref] )
    lab2cm_ref= masses[t_ref]/(masses[p_ref]+masses[t_ref])
    print('Lab2cm: in =',lab2cm_in,'  ref =',lab2cm_ref,'for',p_ref,'+',t_ref,'with Q from ref',Qvalue_ref)
    
# convert to mb for rflow:
    xs_scale  =  xscales[iscale]
  
    data=[]
    angles = set()
    energies = set()
    npts = 0
    for line in d:
        datum = [float(x) for x in line.split()]

        if len(datum)==0: continue
        if elim>0. and float(datum[0])>elim: continue
        #print datum
        datum[0] *= Ein_scale

#         for i in range(4):
#             datum[i] = float('%15.6e' % datum[i])
                    
        # now convert to projectile Elab in Projectile channel. Need Q value.

        datum.append( (datum[0]*lab2cm_in + Qvalue_ref)/lab2cm_ref )   # add lab projectile energy in ref partition : datum[4]
        
        
        datum[2] *= xs_scale
        datum[3] *= xs_scale
#         if staterror>0: datum[3] = max(datum[2]*staterror,datum[3])
        if staterror>0 and datum[3]==0.: datum[3] = datum[2]*staterror
        data.append(datum)
        energies.add(datum[0])   
        a = datum[1]
        if Aflip: a = 180-a
        angles.add(a)
        npts += 1
    #print "\n",data
    anglelist = sorted(angles)
    energylist  = sorted(energies)
    #print ' Angles:',anglelist
    #print ' Energies: ',energylist
    
    if group in ['A','E']:       # use expt prescription
        grp_angles = group == 'A' 
    else:                        # use relative data lengths
        grp_angles = len(anglelist)>len(energylist)
    group = 'A' if grp_angles else 'E'
 
    if grp_angles:  # angular distribution for fixed energies
        type = 0
        collect = 0
        filelist = energylist
        spec = 'energy'
    else:
        type = 2
        collect = 1
        filelist = anglelist
        spec = 'angle'

    varname = spec[0]
    nff = len(filelist)
    print("  # Angles, # Energies: ",len(anglelist),len(energylist), ' out of ',npts,' data points to',nff,', sys,stat error=',syserror,staterror)
    
    iff = 0
    if integrated:   #angle-integrated excitation functions. Make new file with no angle column
        type = 3
        print('  Write 1 file named for angle-integrated cross-section')
        nsf += 1
        splitgroupnorms = False
        base = dr + '-Aint'
        outfile = base + '.data'
        base_v = base
        regex = base + '*'
        reffile = filedir + regex
        ou  = filedir + outfile
        splitdata += ["&Data type=%i pel=%i ic=%i ia=%i data_file='%s' idir=%s iscale=%s abserr=%s / ! %i points" % (type,pel,ic,ia,ou,idir,iscale,abserr,npts) ]
        collect = 1  # column with angle=0, to be removed
        if args.Sfresco: o = open(Dir + outfile,'w')
        pts = 0
        datakeep = []
        for datum in data:
            datum[1] = -1   # indicated angle-integrated data
            if datum[3]==0.0: 
                z_errors.add('File %s excludes zero-error data! Changed to %.1f%%' % (base_v,stat_percent ))
                datum[3] = datum[2]*stat_percent/100.
                
            if Sfactor:
                E_lab = datum[0]
                Ecm = E_lab * masses[t]/(masses[t] + masses[p])
                rmass = masses[p] * masses[t] / (masses[p] + masses[t])
                k = math.sqrt(fmscal * rmass * Ecm)
                eta = etacns * charges[p] * charges[t] * math.sqrt(rmass/Ecm)
                Sfactor_xs = math.exp(-2*pi*eta)/Ecm
                ex2cm = Sfactor_xs   # MeV.b is default
#                 print('Ecm,k,eta,Acm,S =',Ecm,k,eta,datum[1],Sfactor_xs)
            else:
                ex2cm = 1.0
            
            print("%-20s %-10s  %s %s %s %s  %-10s %-10s %s %s %s %s %s" % (datum[4],datum[1],pn,tn,en,rn, 
                    datum[2],datum[3],ex2cm, base_v,'I',datum[0],datum[1]), file=output)
            dat = copy.copy(datum)
            del dat[collect]
            datakeep += [tuple(dat)]
            pts += 1
        for datum in sorted(datakeep, key=lambda x: x[0]):
            if args.Sfresco: print("%s %s %s" % datum[:3], file=o)
        iff += 1     
        parsplit += 1
        
    elif nff > npts//3 and group not in ['A','E']:  # try to split if permitted!
        type = 1
        nsf += 1
        splitgroupnorms = False
        base = dr + '-c'
        base_v = base
        outfile = base + '.data'
        regex = base + '*'
        reffile = filedir + regex
        ou  = filedir + outfile
        print("  Keep original data ",datFile,"in",outfile,"as type=1, because %i > %i or group='%s'" % (nff,npts//3,group))
        splitdata += ["&Data type=%i pel=%i ic=%i ia=%i data_file='%s' idir=%s iscale=%s abserr=%s / ! %i points" % (type,pel,ic,ia,ou,idir,iscale,abserr,npts) ]
        if args.Sfresco: o = open(Dir + outfile,'w')
        pts = 0

        for datum in sorted(data, key=lambda x: x[0]):
            if datum[3]==0.0: 
                z_errors.add('File %s excludes zero-error data! Changed to %.1f%%' % (base_v,stat_percent ))
                datum[3] = datum[2]*stat_percent/100.
            if not lab:
                angle_ex = datum[1]
                ex2cm = 1.0
                Ecm = datum[0]
                mu_cm = math.cos(dat[1]*rad)
            else:
                E_lab = datum[0]
                Ecm = E_lab * masses[t]/(masses[t] + masses[p])
                Q   =  Qvalue
                angle_lab = datum[1]
                mu_lab = math.cos(angle_lab*rad)
                mu_cm,frame_scale = lab2cm(mu_lab, masses[p],masses[t],masses[e],masses[r], E_lab,Q)    
                angle_cm = math.acos(mu_cm)/rad
                datum[1] = angle_cm
                if Aflip: datum[1] = 180 - datum[1]
                angle_ex = angle_lab
                ex2cm = frame_scale if not rRuth else 1.0
                
            if rRuth:
                rmass = masses[p] * masses[t] / (masses[p] + masses[t])
                k = math.sqrt(fmscal * rmass * Ecm)
                eta = etacns * charges[p] * charges[t] * math.sqrt(rmass/Ecm)
                Rutherford = (eta/(k*(1-mu_cm)) )**2 * 10.  # mb
                ex2cm = Rutherford  # convert ratio-to-Rutherford to cm barns
                    
            print("%-20s %-10s  %s %s %s %s  %-10s %-10s %s %s %s %s %s" % (datum[4],datum[1],pn,tn,en,rn,  
                    datum[2],datum[3],ex2cm, base_v,'N',datum[0],angle_ex), file=output)
            if args.Sfresco:print("%s %s %s %s" % tuple(datum[:4]), file=o)
            pts += 1
        iff += 1
        parsplit += 1
        
    else:
        nsf1 = nsf
        print('  Write ',nff,' files (',nsf+1,'to',nsf+nff,') named for ',spec,'=',filelist,' as ',group)
        parsplit1 = parsplit+1
        base = dr + ('f' if Aflip else '') + '@' + varname
        regex = base + '*'
        reffile = filedir + regex
            
        for var in filelist:
            parsplit += 1
            # print 'File named for ',var,', splitgroupnorms=',splitgroupnorms
            base_v = base + str(var) 
            outfile =  base_v + '.data'
            if args.Sfresco: o = open(Dir + outfile,'w')
            pts = 0
            datakeep = []
            for datum in data:
                dat = copy.copy(datum)
                if Aflip and not lab: dat[1] = 180 - dat[1]

                if var==dat[collect]:
                    if dat[3]==0.0: 
#                       z_errors.add('File %s excludes zero-error data! Changed to %.1f%%' % (base_v,stat_percent ))
                        z_errors.add('File %s excludes zero-error data! Changed to %s%%' % (base_v,stat_percent ))
                        continue
                        print('stat_percent=',stat_percent)
                        print('stat_percent type=',type(stat_percent))
                        print('dat=',dat,type(dat),len(dat))
                        print('dat[2]=',dat[2],type(dat[2]))
                        print('dat[2]*stat_percent=',dat[2]*stat_percent)
                        dat[3] = dat[2]*stat_percent/100.
                    
                    if not lab:
                        angle_ex = dat[1]
                        ex2cm = 1.0
                        Ecm = dat[0] * masses[t]/(masses[t] + masses[p])
                        mu_cm = math.cos(dat[1]*rad)
                    else:
                        E_lab = dat[0]
                        Ecm = E_lab * masses[t]/(masses[t] + masses[p])
                        Q   =  Qvalue
                        angle_lab = datum[1]
                        mu_lab = math.cos(angle_lab*rad)
                        mu_cm,frame_scale = lab2cm(mu_lab, masses[p],masses[t],masses[e],masses[r], E_lab,Q)    
                        angle_cm = math.acos(mu_cm)/rad
                        
                        dat[1] = angle_cm
                        if Aflip: dat[1] = 180 - dat[1]
                        angle_ex = angle_lab
                        ex2cm = frame_scale if not rRuth else 1.0
                         
                    if rRuth:
                        rmass = masses[p] * masses[t] / (masses[p] + masses[t])
                        k = math.sqrt(fmscal * rmass * Ecm)
                        eta = etacns * charges[p] * charges[t] * math.sqrt(rmass/Ecm)
                        Rutherford = (eta/(k*(1-mu_cm)) )**2 * 10.   # mb
                        ex2cm = Rutherford
#                         print('Ecm,k,eta,Acm,R =',Ecm,k,eta,dat[1],Rutherford)
                              
                    print("%-20s %-10s  %s %s %s %s  %-10s %-10s %s %s %s %s %s" % (dat[4],dat[1],pn,tn,en,rn, 
                            dat[2],dat[3],ex2cm, base_v,group,dat[0],angle_ex), file=output)
                    del dat[collect]
                    #print >>t,dat
                    #print >>o,"%s %s %s" % tuple(dat)
                    datakeep += [tuple(dat)]
                    pts += 1
            for datum in sorted(datakeep, key=lambda x: x[0]):
                if args.Sfresco: print("%s %s %s" % datum[:3], file=o)
            iff += 1

            ou  = filedir + outfile
            Mflip = Aflip and lab
            splitdata += ["&Data type=%i pel=%i ic=%i ia=%i %s=%s data_file='%s' idir=%s iscale=%s lab=%s Mflip=%s abserr=%s / ! %i points in %i/%i :: set %i" % (type,pel,ic,ia,spec,var,ou,idir,iscale,lab,Mflip,abserr,pts,iff,nff,parsplit)]
            nsf += 1

            if splitgroupnorms:  # make renormalization variables for each group in split
                name='n:' + outfile.split('.data')[0]
                norm = norm_adj.get(ou,expect)
                print("For output file",ou,"get norm=",norm," in adj:",ou in list(norm_adj.keys()))
                datanorms += ["&Variable kind=5 name='%s'  datanorm=%s step=0.01   reffile='%s'/ " % (name,norm,ou)] 

                if syserror>0: 
                    normlimits += ["&Data type=6 value=%s error=%s abserr=T   reffile='%s' kind=%i/" % (expect,syserror,ou,5)]
                    flownorms +=["%-10s %-10s  %-20s %-10s %-10s %s" % (norm, 0.01, name, expect,syserror,  base_v) ]
                else:
                    flownorms +=["%-10s %-10s  %-20s %-10s %-10s %s" % (norm, 0.01, name, 0,0,  base_v) ]

            if splitgroupshifts:  # make energy shift variables for each group in split
                name='s:' + outfile.split('.data')[0]
                shift = shift_adj.get(ou,eshift)
                datashifts +=["&Variable kind=6 name='%s'  dataEshift=%s step=%s   reffile='%s'/ " % (name,shift,ecalib,ou)]

                if ecalib>0: shiftlimits += ["&Data type=6 value=%s error=%s abserr=T   reffile='%s' kind=%i/" % (eshift,ecalib,ou,6)]

    if not splitgroupnorms:
        name='r:' + baseFile.split('.dat')[0]
        norm = norm_adj.get(reffile,expect)
        datanorms +=["&Variable kind=5 name='%s'  datanorm=%s step=0.01   reffile='%s'/ " % (name,norm,reffile)]

        if syserror>0: 
            normlimits += ["&Data type=6 value=%s error=%s abserr=T   reffile='%s'  kind=%i/" % (expect,syserror,reffile,5)]
            flownorms +=["%-10s %-10s  %-20s %-10s %-10s %s" % (norm, 0.01, name, expect,syserror, regex ) ]
        else:
            flownorms +=["%-10s %-10s  %-20s %-10s %-10s %s" % (norm, 0.01, name, 0,0, regex,) ]
                            
    if not splitgroupshifts:
        name='S:' + datFile.split('.dat')[0]
        if eshift !=0.0  or ecalib !=0.:
            shift = shift_adj.get(reffile,eshift)
            datashifts +=["&Variable kind=6 name='%s' dataEshift=%s step=%s reffile='%s' /" % (name,shift,ecalib,reffile)]

        if ecalib > 0.0: shiftlimits += ["&Data type=6 value=%s error=%s abserr=T   reffile='%s' kind=%i/" % (eshift,ecalib,reffile,6)]
            
    flast = parsplit
    print("plot data_%s_%i-%i.plot %i %i" % (datFile,ffirst,flast,ffirst,flast), file=plotd)
    ffirst = flast+1
#
#  Print other &Variable and &Data namelists for split data
#
for  s in flownorms: print(s, file=noutput)

print('Zero-error cases\n',z_errors)
for ze in sorted(list(z_errors)):
    print(ze, file=z_errorOut)

if args.Sfresco:

    for  s in datanorms: print(s, file=sf)
    print(' ', file=sf)
    for  s in datashifts: print(s, file=sf)

    print(' ', file=sf)

    for  s in splitdata: print(s, file=sf)
    print(' ', file=sf)
    for  s in normlimits: print(s, file=sf)
    print(' ', file=sf)
    for  s in shiftlimits: print(s, file=sf)
