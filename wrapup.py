
##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

import os,math,numpy,cmath,pwd,sys,time,json

from xData.Documentation import documentation as documentationModule
from xData.Documentation import computerCode  as computerCodeModule
from functools import singledispatch

plcolor = {0:"black", 1:"red", 2:"green", 3: "blue", 4:"yellow", 5:"brown", 6: "grey", 7:"violet",
            8:"cyan", 9:"magenta", 10:"orange", 11:"indigo", 12:"maroon", 13:"turquoise", 14:"darkgreen"}
pldashes = {0:'solid', 1:'dashed', 2:'dashdot', 3:'dotted'}
plsymbol = {0:".", 1:"o", 2:"s", 3: "D", 4:"^", 5:"<", 6: "v", 7:">",
            8:"P", 9:"x", 10:"*", 11:"p", 12:"D", 13:"P", 14:"X"}

lightnuclei = {'n':'n', 'H1':'p', 'H2':'d', 'H3':'t', 'He3':'h', 'He4':'a', 'photon':'g'}

from PoPs.chemicalElements.misc import *
from xData import date

def now():
    return date.Date( resolution = date.Resolution.time )

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

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(numpy.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return numpy.float64(val)

def saveNorms2gnds(gnd,docData,previousFit,computerCodeFit,inFile,n_norms,norm_val,norm_refs):

    docLines = ['Rflow:']
    for n in range(n_norms):
        docLines.append("&variable name='%s' kind=5 dataset=0, 0 datanorm=%12.5e step=0.01, reffile='%s'/" % ( norm_refs[n][0] ,norm_val[n], norm_refs[n][1]) ) 
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
        computerCodeFit = computerCodeModule.ComputerCode( label = 'R-matrix fit', name = 'Rflow', version = '') #, date = now() )
        deckLabel = 'Fitted_data'
        
    inputDataSpecs = computerCodeModule.InputDeck( deckLabel , inFile, ('\n  %s\n' % now() )  + ('\n'.join( docLines ))+'\n' )
    computerCodeFit.inputDecks.add( inputDataSpecs )

    if not previousFit: 
        RMatrix = gnd.resonances.resolved.evaluated    
        RMatrix.documentation.computerCodes.add( computerCodeFit )
    return

def plotOut(n_data,n_norms,n_dof,args, base,info,dataDir, inclusiveCaptures,
    chisqTOT,ww,data_val,norm_val,norm_info,effect_norm,norm_refs, previousFit,computerCodeFit,
    groups,cluster_list,group_list,Ein_list,Aex_list,xsc,X4groups, data_p,pins, TransitionMatrix,
    XCLUDE,p,projectile4LabEnergies,data_lines,dataFile,
    EIndex,totals,pname,tname,datasize, ipair,cm2lab, emin,emax,pnin,gnd,cmd ):

    print('\nExperimental data groups, by energy or angle groups if simpler:\n')
    Matplot = True
    
    ngraphAll = 0
    groups = sorted(groups)
    chisqAll = 0
#     plot_cmds = []
    plot_cmd = 'xmgr '
    worse = []
    unadjustedShapes = {}
    data_filtered = []
    for group in groups:

        found = False
        for id in range(n_data):
            if group == group_list[id] and cluster_list[id]!='I': found = True    # not duplicate of later

        if args.Cross_Sections and found:
            g_out = group+info+'-fit'
            if '/' in g_out:  g_out =  g_out.split('/')[1].replace('/','+')
            g_out = dataDir + '/' + g_out
            gf = open(g_out,'w')
            e_out = group+info+'-expt'
            if '/' in e_out: e_out = e_out.split('/')[1].replace('/','+')
            e_out = dataDir + '/' + e_out
            ef = open(e_out,'w')
            op = ' in file %-43s' %  g_out
        else:
            op = ' in group %-43s' %  group
        ope = op if '/' not in op else op.split('/')[1]

        ie = 0
        io = 0
        chisq = 0.0
        for id in range(n_data):
            if group == group_list[id]:
                gr = group
                if '/' in gr: gr = gr.split('/')[1]
                fac = 1.0
                unadjustedShape = False
                if not args.norm1:
                    for ni in range(n_norms):
                        fac += (norm_val[ni]-1.) * effect_norm[id,ni]
                        unadjustedShape  = unadjustedShape or (effect_norm[id,ni] > 0. and norm_info[ni,1] == 0.0 and norm_val[ni] == 1.0)  # fitted norm still original 1.0
#                             print('effect_norm[',id,ni,']',effect_norm[id,ni],norm_info[ni,0],norm_info[ni,1] )
                if unadjustedShape:
                    if not unadjustedShapes.get(gr,False): print('\n    *** Unadjusted shape data:',gr,' -   exclude from chi^2 sum\n')
                    unadjustedShapes[gr] = True
                    continue
                Data = data_val[id,2]*fac
                DataErr = data_val[id,3]*fac  
                ex2cm = data_val[id,4] 
                chi = (xsc[id]/fac/ex2cm-data_val[id,2])/data_val[id,3] 
                
                if args.Cross_Sections and found:
                    cluster = cluster_list[id]
                    Ein = Ein_list[id]
                    Aex = Aex_list[id]
                    if cluster == 'A':
#                         theta = math.acos(data_val[id,1])*180./pi
                        print(Aex, xsc[id]/ex2cm, chi, file=gf)
                        print(Aex, Data, DataErr, file=ef)                   
                    elif cluster in ['E','I']:
                        print(Ein, xsc[id]/ex2cm, 'chi=',chi, 'd0=',data_val[id,0],id,file=gf)
                        # print(data_val[id,0], xsc[id]/ex2cm, 'chi=',chi, 'Ein',Ein,id,file=gf)
                        print(Ein, Data, DataErr, file=ef)                                
                    else:  # cluster == 'N':  xyz
#                         theta = math.acos(data_val[id,1])*180./pi
                        print(Ein, Aex, xsc[id]/ex2cm, chi, file=gf)   # xyz+chi
                        print(Ein, Aex, Data, DataErr, file=ef)  #xyzdz 
                
                chisq += chi**2
                io += 1
                if XCLUDE is not None and abs(chi) < XCLUDE:
                    data_filtered.append(data_lines[id])
            ie += 1
        if args.Cross_Sections and found:
            gf.close()
            ef.close()
        print('Model %2i curve (%4i pts)  %-50s:   chisq/gp =%9.3f  %8.3f %%' % (ngraphAll+1,io,ope,chisq/max(1,io),chisq/chisqTOT*100.) )

        if not unadjustedShapes.get(gr,False): worse.append([chisq/chisqTOT*100.,gr])
        
        if args.Cross_Sections and found: 
            plot_cmd += ' -graph %i -xy %s -xydy %s ' % (ngraphAll,g_out,e_out) 
#             plot_cmds.append(plot_cmd)
        chisqAll += chisq
        ngraphAll += 1
        
    if XCLUDE is not None:
        data_new_name = dataFile.replace('.','_n.') + '_X%s' % XCLUDE
        with open(data_new_name,'w') as fout: 
            fout.writelines([projectile4LabEnergies+'\n'] + data_filtered)
        print('Filtered data file',data_new_name,'with',len(data_filtered),' points as |chi| < %.2f' % XCLUDE)
            
#     plot_cmds.append(plot_cmd)
# chi from norm_vals themselves:
    for ni in range(n_norms):
        chi = (norm_val[ni] - norm_info[ni,0]) * norm_info[ni,1]
        chisq = chi**2   
        chi_note = ' shape' if norm_info[ni,1] == 0.0 else ''
        if norm_val[ni] == 1.0: chi_note += ' unscaled'
        print('Norm scale   %10.6f         %-30s ~ %10.5f :    chisq    =%9.3f  %8.3f %%  %s' % (norm_val[ni] , norm_refs[ni][0],norm_info[ni,0], chisq,chisq/chisqTOT*100.,chi_note) )
        gr = norm_refs[ni][0]
        if '/' in gr: gr = gr.split('/')[1]
#         if norm_info[ni,1] > 0.0: 
        worse.append([chisq/chisqTOT*100., gr])
        chisqAll += chisq
    print('\n Last chisq/pt  = %10.5f from %i points' % (chisqAll/max(1,n_data),n_data),' so chisq/dof = %10.5f' % (chisqAll/n_dof), '(dof =',n_dof,')\n' )   
#     if args.Cross_Sections and ngraphAll < 20: 
#          print("Plot with\n",plot_cmd,'\n with required rows and cols for',ngraphAll,'graphs')
    
#     for plot_cmd in plot_cmds: print("Plot:    ",plot_cmd)

#     with open('data.out','w') as fout:
#         for id in range(n_data):
#             pin,pout = data_p[id,:]
#             print(id,data_val[id,:],'for',pin,pout,':', totals[pout,pin,EIndex[id]] , file=fout)
    
    X4groups = sorted(X4groups)
#     print('\nX4groups sorted:',X4groups)
#     if len(X4groups)<1: sys.exit()
    print('Data grouped by X4 subentry:')
    chisqAll = 0
    plot_cmds = []
    legendsize =  0.5 # default
    info = info.replace('_','')
    
    for group in X4groups:
        groupB = group.split('@')[0]
        ng = 0
        plot_cmd = ''
        ngraphAll = 0
        if args.Cross_Sections:
            g_out = group+info+'-fit'
            if '/' in g_out: g_out = g_out.split('/')[1].replace('/','+')
            g_out = dataDir + '/' + g_out
            gf = open(g_out,'w')
            e_out = group+info+'-expt'
            if '/' in e_out: e_out = e_out.split('/')[1].replace('/','+')
            e_out = dataDir + '/' + e_out
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
        xlabel = 'Energy [MeV, lab]'  # default
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
                pn = quickName(pname[pin],tname[pin])
                if pout == -1:
                    reaction = '%s total' % pn
                elif pout == -2:
                    reaction = '%s capture' % pn
                elif pout == pin:
                    reaction = '%s elastic' % pn
                else:
                    reaction = pins[pin]+'->'+pins[pout]
                    
                fac = 1.0 # assume same for all data(id) in this curve!
                shape = False
                if not args.norm1:
                    for ni in range(n_norms):
                        fac += (norm_val[ni]-1.) * effect_norm[id,ni]
                        shape = shape or norm_info[ni,1]==0.0
                lfac = (fac-1)*100
                curves.add((curve,fac,lfac,shape,pin,pout,reaction))
     
        if args.debug: print('\nGroup',group,'has curves:',curves)
        ncurve = 0
        for curve,fac,lfac,shape,pin,pout,reaction in curves:
            pn = quickName(pname[pin],tname[pin])
            if pout == -1:
                po = 'tot'
            elif pout == -2:
                po = 'cap'
            elif pout == pin:
                po = 'el'
            else:
                po = quickName(pname[pout],tname[pout])
                    
            tag = groupB + ( ('@' + curve) if curve!='Aint' else '')
            if ptsInCurve[curve]==0: continue
#             print('\nCurve for ',tag)
            
            lchisq = 0.
            if Matplot:
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
                xlabel = 'Incident %s energy [MeV, lab]' % pn if cluster in ['E','I'] else 'Scattering angle [deg, cm]'
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
                if Matplot:    
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
                
            if Matplot:    # end of curve
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
                    'symbol': plsymbol[ng%13], 'symbolsize':args.datasize   }
                LineModel[0] = {'kind':'Model', 'color':plcolor[ic-1], 'linestyle': pldashes[(ng-1)%4], 'evaluation':''} 
#                 DataLines.append(LineData)
#                 ModelLines.append(LineModel)
                if args.debug: print('Finishing new curve',ng,'for',curve,'with legend',legend,'with',ptsInCurve[curve],'pts')

                DataLines.append(LineData)
                ModelLines.append(LineModel)
        
        ncurve += 1
        ope = op if '/' not in op else op.split('/')[1]
        print('Model %2i %2i curves (%4i pts)  %-50s:   chisq/gp =%9.3f  %8.3f %%' % (ncurve,ng,io,ope,chisq/max(1,io),chisq/chisqTOT*100.) )
        chisqAll += chisq
        ngraphAll += 1
        
        if args.Cross_Sections:    # wrap up this subentry
            gf.close()
            ef.close()
            plot_cmd += 'xmgr -xy %s -xydy %s ' % (g_out,e_out) 
            plot_cmds.append(plot_cmd)

        if Matplot:           # wrap up this subentry
            subtitle = "Using " + args.inFile + ' with  '+args.dataFile+" & "+args.normFile + ', Chisq/pt =%.3f' % (chisq/io)
            kind     = 'R-matrix fit of '+group.split('@')[0]+' for '+reaction+' (units mb and MeV)'
            GraphList.append([DataLines+ModelLines,subtitle,args.logs,kind])

            j_out = group+info+'_%s-%s.json' % (pn,po)
            if '/' in j_out: j_out = j_out.split('/')[1].replace('/','+')
            j_out = dataDir + '/' + j_out
            with open(j_out,'w') as ofile:
               json.dump([1,1,[cmd,xlabel],GraphList],ofile, default=to_serializable)
               
            plot_cmd += '\t             json2pyplot.py -w 10,8 %s' % j_out
            plot_cmds.append(plot_cmd)
#
# Extract newest R-matrix parameters

    rrr = gnd.resonances.resolved
    RMatrix = rrr.evaluated
    bndx = RMatrix.boundaryCondition
    pLab = quickName(pname[ipair],tname[ipair])
    PoleData = []    
    PoleDataP = []    
    ModelLines = []
    jset = 0
    PoleGraphList = []
    ngraphAll = 0
    print('Kept data in cm range [',emin,',',emax,'] for projectile',pname[ipair])

    for Jpi in RMatrix.spinGroups:   # jset values need to be in the same order as before
        J_set = Jpi.spin
        pi_set = Jpi.parity
        parity = '+' if int(pi_set) > 0 else '-'
        legend = '%s%s' % (J_set,parity)
        pcolor = 'red' if int(pi_set) > 0 else 'black'
#             if True: print('J,pi =',J_set,parity)
        R = Jpi.resonanceParameters.table
        rows = R.nRows
        cols = R.nColumns - 1  # without energy col
        LineData  = [{}, [[],[],[],[]] ]
        LineDataP  = [{}, [[],[],[],[]] ]
        LineModel = [{}, [[],[],[],[]] ]
        bit = 0.05 if int(pi_set) > 0 else 0.0
        for pole in range(rows):
            E_pole = R.data[pole][0]      
            E_cm = E_pole / cm2lab[ipair]
            if not (emin < E_cm < emax): continue
            LineData[1][0].append(E_cm)
            LineData[1][1].append(float(J_set) + bit)
            LineData[1][2].append(0)
            LineData[1][3].append(0)  
                 
            LineDataP[1][0].append(E_cm)
            LineDataP[1][1].append(float(J_set) + bit)
            LineDataP[1][2].append(0)
            LineDataP[1][3].append(0)   
                                
        LineData[0] =   {'kind':'Data',  'color':pcolor, 'capsize':0.10, 'legend':legend, 'legendsize':legendsize,
            'symbol': plsymbol[jset%13], 'symbolsize':datasize   }
        LineDataP[0] =  {'kind':'Data',  'color':pcolor, 'capsize':0.10, 'legend':legend, 'legendsize':legendsize,
            'symbol': plsymbol[jset%13], 'symbolsize':datasize   }
        jset += 1
        PoleData.append(LineData)
        PoleDataP.append(LineDataP)
        
    ModelLines.append(LineModel)
    subtitle = '' # "Using " + args.inFile + ' with  '+args.dataFile+" & "+args.normFile
    kind     = "Final Pole Energies (MeV, %s cm) in B=%s basis" % (pname[ipair],bndx)
    PoleGraphList.append([PoleData+ModelLines,subtitle,args.logs,kind])
    p_out = 'Pole-energies.json' 
    if totals is not None: p_out = dataDir + '/' + p_out  # put in subdirectory if it exists
    print('   Write',p_out,'with',1)
    with open(p_out,'w') as ofile:
       json.dump([1,1,[cmd,xlabel],PoleGraphList],ofile, default=to_serializable)
            
# chi from norm_vals themselves:
    for ni in range(n_norms):
        chi = (norm_val[ni] - norm_info[ni,0]) * norm_info[ni,1]
        chi_note = ' shape' if norm_info[ni,1] == 0.0 else ''
        if norm_val[ni] == 1.0: chi_note += ' unscaled'
        print('Norm scale   %10.6f         %-30s ~ %10.5f :     chisq    =%9.3f  %8.3f %% %s' % (norm_val[ni] , norm_refs[ni][0],norm_info[ni,0], chi**2, chi**2/chisqTOT*100.,chi_note) )
        chisqAll += chi**2
#     print('\n Last chisq/pt  = %10.5f from %i points' % (chisqAll/max(1,n_data),n_data),' so chisq/dof = %10.5f' % (chisqAll/n_dof), '(dof =',n_dof,')' )   
    
#     for plot_cmd in plot_cmds: print("Plot:    ",plot_cmd)

    if totals is not None:
        print('\nWrite Angle-integrals-*.json files in the lab frame of gnds projectile')
        Singleplot_cmds = ''
        plot_cmds = ''
        Globalplot_cmds = ''
        npairs = totals.shape[1]
        nmodelpts = data_val.shape[0]
        GlobalGraphList = []
#         GlobalGraphList.append(PoleGraphList[0])
        GlobalngraphAll = 0 # 1 # this is the PoleGraph
        pLab = quickName(pname[ipair],tname[ipair])
        lab2cm = 1./cm2lab[ipair]
      
        for pinG in range(npairs):
            pn = quickName(pname[pinG],tname[pinG])
            print('In:',pn,'from',pinG)
            tnucl,tlevel = nuclIDs(tname[pinG])
            if tlevel>0: continue  # don't bother with initial excited states.
            GraphList = []
            ngraphAll = 0

            for poutG in range(npairs+1):
                if pinG==poutG: continue  # elastic is too boring
                po = '-> ' + quickName(pname[poutG],tname[poutG]) if poutG < npairs else 'tot'
#                 po = '-> ' + pname[poutG] if poutG < npairs else 'tot'
                poG = poutG if poutG < npairs else -1  # convention in printExcitationFunctions
                SingleGraphList = []
                SinglengraphAll = 0
                          
                DataLines = []
                dataPoints = 0
                ModelLines = []
                LineModel = [{}, [[],[],[],[]] ]
                for i0 in range(nmodelpts):
                    i = EIndex[i0]
                    if totals[poG,pinG,i] < 0.0: continue
#                     if pinG != data_p[i,0] or poG != data_p[i,1] : continue
                    LineModel[1][0].append(data_val[i,0]*lab2cm)
                    LineModel[1][1].append(totals[poG,pinG,i])
                    LineModel[1][2].append(0.)
                    LineModel[1][3].append(0.)    
                legend = '%s %s' % (pn,po) #+ ' (%s,%s=%s)' % (pinG,poutG,poG)
                LineModel[0] = {'kind':'Model', 'color':'orange', 'linestyle': 'solid', 'evaluation':legend}  #, 'legend': legend } 
                ModelLines.append(LineModel)
                print('  Curve for',legend)
                
                ng = 0
                for group in X4groups:
#                     print('  X4 group:',group)
                    groupB = group.split('@')[0]
                    lfac = 1.0
                    curves = set()
                    ptsInCurve = {}  # to be list of curves for given group base (groupB)
                    for id in range(n_data):
                        gld = group_list[id]
                        glds= gld.split('@')
                        if groupB == glds[0]:
                            curve = glds[1] if len(glds)>1 else 'Aint'
                            ptsInCurve[curve] = ptsInCurve.get(curve,0) + 1
                
                            pin,pout = data_p[id,:]   # assume same for all data(id) in this curve!
                            if pin != pinG or pout != poG: continue
                            if pout == -1:
                                reaction = '%s total' % pn
                            elif pout == pin:
                                reaction = '%s elastic' % pn
                            else:
                                reaction = pins[pin]+'->'+pins[pout]
                    
                            fac = 1.0 # assume same for all data(id) in this curve!
                            shape = False
                            if not args.norm1:
                                for ni in range(n_norms):
                                    fac += (norm_val[ni]-1.) * effect_norm[id,ni]
                                    shape = shape or norm_info[ni,1]==0.0
                            lfac = (fac-1)*100
                            curves.add((curve,fac,lfac,shape,pin,pout,reaction))
#                     if len(curves)>0: print('\nGroup',group,'has curves:',curves)     
                    
                    for curve,fac,lfac,shape,pin,pout,reaction in curves:
                        if pin != pinG or pout != poG: continue
                        if ptsInCurve[curve]==0: continue
#                         print('Data for',legend,'from',curve)
            
                        tag = groupB + ( ('@' + curve) if curve!='Aint' else '')
            
                        LineData  = [{}, [[],[],[],[]] ]
                        np = 0
                        for id in range(n_data): # range(Aintrange[0],Aintrange[1]): # range(n_data):  # # n_angle_integrals0:n_totals0
                            gld = group_list[id]
                            cluster = cluster_list[id]
                            if gld != tag: continue
                            if cluster != 'I': continue # angle-integrated data only here
#                             if pin != data_p[id,0] or pout != data_p[id,1] : continue
                            E_Gproj = data_val[id,0]*lab2cm
                            Data = data_val[id,2]*fac
                            DataErr = data_val[id,3]*fac        

                            LineData[1][0].append(E_Gproj)
                            LineData[1][1].append(Data)
                            LineData[1][2].append(0)
                            LineData[1][3].append(DataErr)
                            np += 1
                            dataPoints += 1
                        if np>0: 
                            print('    Curve',ng,':',tag,'has',np,'data points')
                                
                            ng += 1
                            ic = (ng+ npairs-1) % 14 + 1  # for colors
                            leg = tag if '/' not in tag else tag.split('/')[1]
                            legend = leg.replace('-Aint','') + ' * %.2f' % fac # if len(leg) < 12 else leg[:12]
                            LineData[0] =  {'kind':'Data',  'color':plcolor[ic-1], 'capsize':0.10, 'legend':legend, 'legendsize':legendsize,
                                'symbol': plsymbol[ng%13], 'symbolsize':datasize   }
    #                         print('    Finishing I curve',ng,'for',curve,'with legend',legend,'with',ptsInCurve[curve],'pts for ',reaction)

                            DataLines.append(LineData)
                selection = 'all' if TransitionMatrix<=0 else '(>= %i datapt)' % TransitionMatrix
                subtitle = '' # "Using " + args.inFile + ' with  '+args.dataFile+" & "+args.normFile
                kind     = "R-matrix fit for incident %s+%s  (%s)  (units mb and %s MeV cm)" % (pname[pinG],tname[pinG],pn,pname[ipair])
                kinds    = "R-matrix fit for incident %s+%s %s  (units mb and %s MeV cm)" % (pname[pinG],tname[pinG],po,pname[ipair])
                kindG    = "R-matrix fits for %s transitions (units mb and %s MeV cm)" % (selection,pname[ipair])
                SingleGraphList.append([DataLines+ModelLines+PoleDataP,subtitle,args.logs,kinds])
                SinglengraphAll += 1

                if dataPoints >= TransitionMatrix:      # skip graph placement in summary plots, if not enough data. 
                    GraphList.append([DataLines+ModelLines,subtitle,args.logs,kind])
                    ngraphAll += 1
                    GlobalGraphList.append([DataLines+ModelLines,subtitle,args.logs,kindG])
                    GlobalngraphAll += 1

                j_out = 'Angle-integrals-%s-to-%s.json' % (pn,po.replace('-> ',''))
        #         if '/' in j_out: j_out = j_out.split('/')[1].replace('/','+')
                j_out = dataDir + '/' + j_out
#                 print('Write',j_out,'with',SinglengraphAll)
                with open(j_out,'w') as ofile:
                   json.dump([SinglengraphAll,1,[cmd,xlabel],SingleGraphList],ofile, default=to_serializable)
                Singleplot_cmds +=  ' ' + j_out
        
            j_out = 'Angle-integrals-from-%s.json' % (pn)
    #         if '/' in j_out: j_out = j_out.split('/')[1].replace('/','+')
            j_out = dataDir + '/' + j_out
            print('   Write',j_out,'with',ngraphAll)
            with open(j_out,'w') as ofile:
               json.dump([ngraphAll,1,[cmd,xlabel],GraphList],ofile, default=to_serializable)
            plot_cmds += ' ' + j_out
        
        j_out = 'Angle-integrals-Global.json' 
#         if '/' in j_out: j_out = j_out.split('/')[1].replace('/','+')
        j_out = dataDir + '/' + j_out
        print('Write',j_out,'with',GlobalngraphAll)
        with open(j_out,'w') as ofile:
           json.dump([GlobalngraphAll,1,[cmd,xlabel],GlobalGraphList],ofile, default=to_serializable)
        Globalplot_cmds +=  ' ' + j_out
        
        print('------')
#         print("Single plots:   plotJson.py -w 10,8 ",Singleplot_cmds)
#         print("Incident plots: plotJson.py -w 10,8 ",plot_cmds)
#         print("Global plots:   plotJson.py -w 10,8 ",Globalplot_cmds)
#     print(    "Poles E plot:   plotJson.py -w 10,8 ",p_out)
    print("\nWorst fits:")
    worse.sort(key = lambda x: -x[0])
    for i in range(min(10,len(worse))):
        bad = worse[i]
        print('    %-40s contributes %9.3f %%' % (bad[1],bad[0]))
    print("\nUnadjusted shape data:")
    for gr in unadjustedShapes.keys():
        print('    ',gr)
    print('------')

    return
