import os,math,numpy,cmath,pwd,sys,time,json

from xData.Documentation import documentation as documentationModule
from xData.Documentation import computerCode  as computerCodeModule
from functools import singledispatch

plcolor = {0:"black", 1:"red", 2:"green", 3: "blue", 4:"yellow", 5:"brown", 6: "grey", 7:"violet",
            8:"cyan", 9:"magenta", 10:"orange", 11:"indigo", 12:"maroon", 13:"turquoise", 14:"darkgreen"}
pldashes = {0:'solid', 1:'dashed', 2:'dashdot', 3:'dotted'}
plsymbol = {0:".", 1:"o", 2:"s", 3: "D", 4:"^", 5:"<", 6: "v", 7:">",
            8:"P", 9:"x", 10:"*", 11:"p", 12:"1", 13:"2", 14:"3"}

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(numpy.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return numpy.float64(val)

def saveNorms2gnds(gnd,docData,previousFit,computerCodeFit,n_norms,norm_val,norm_refs):

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
        deckLabel = 'Fitted_data'
        
    inputDataSpecs = computerCodeModule.InputDeck( deckLabel , ('\n  %s\n' % time.ctime() )  + ('\n'.join( docLines ))+'\n' )
    computerCodeFit.inputDecks.add( inputDataSpecs )

    if not previousFit: RMatrix.documentation.computerCodes.add( computerCodeFit )
    return

def plotOut(n_data,n_norms,dof,args, base,info,dataDir, chisqtot,data_val,norm_val,norm_info,effect_norm,norm_refs, previousFit,computerCodeFit,
    groups,cluster_list,group_list,Ein_list,Aex_list,xsc,X4groups, data_p,pins, evaluation,cmd ):

    ngraphAll = 0
    groups = sorted(groups)
    chisqAll = 0
#     plot_cmds = []
    plot_cmd = 'xmgr '
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
            ie += 1
        if args.Cross_Sections and found:
            gf.close()
            ef.close()
        print('Model %2i curve (%4i pts)%s:   chisq/gp =%9.3f  %8.3f %%' % (ngraphAll+1,io,op,chisq/io,chisq/chisqtot*100.) )
        if args.Cross_Sections and found: 
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
    print('\n Last chisq/pt  = %10.5f from %i points' % (chisqAll/max(1,n_data),n_data) )  
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
        print('\nModel %2i %2i curves (%4i pts)%s:   chisq/gp =%9.3f  %8.3f %%' % (ncurve,ng,io,op,chisq/max(1,io),chisq/chisqtot*100.) )
        chisqAll += chisq
        ngraphAll += 1
        
        if args.Cross_Sections:    # wrap up this subentry
            gf.close()
            ef.close()
            plot_cmd += 'xmgr -xy %s -xydy %s ' % (g_out,e_out) 
            plot_cmds.append(plot_cmd)

        if args.Matplot:           # wrap up this subentry
            subtitle = "Using " + args.inFile + ' with  '+args.dataFile+" & "+args.normFile + ', Chisq/pt =%.3f' % (chisq/io)
            kind     = 'R-matrix fit of '+group.split('@')[0]+' for '+reaction+' (units mb and MeV)'
            GraphList.append([DataLines+ModelLines,subtitle,args.logs,kind])

            j_out = group+info+'.json'
            if '/' in j_out: j_out = j_out.split('/')[1].replace('/','+')
            j_out = dataDir + '/' + j_out
            with open(j_out,'w') as ofile:
               json.dump([1,1,cmd,GraphList],ofile, default=to_serializable)
               
            plot_cmd += '\t             json2pyplot.py -w 10,8 %s' % j_out
            plot_cmds.append(plot_cmd)

# chi from norm_vals themselves:
    for ni in range(n_norms):
        chi = (norm_val[ni] - norm_info[ni,0]) * norm_info[ni,1]        
        print('Norm scale   %10.6f         %-30s ~ %10.5f :     chisq    =%9.3f  %8.3f %%' % (norm_val[ni] , norm_refs[ni][0],norm_info[ni,0], chi**2, chi**2/chisqtot*100.) )
        chisqAll += chi**2
    print('\n Last chisq/pt  = %10.5f from %i points' % (chisqAll/max(1,n_data),n_data) )  
    print(  ' Last chisq/dof = %10.5f' % (chisqAll/dof), '(dof =',dof,')' )   
    
    for plot_cmd in plot_cmds: print("Plot:    ",plot_cmd)

    return
