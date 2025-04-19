
##############################################
#                                            #
#    Rflow 0.30      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

from collections import OrderedDict
import math,numpy,os,pwd
from pqu import PQU as PQUModule
from fudge import reactionSuite as reactionSuiteModule

from fudge.processing.resonances.getCoulombWavefunctions import *
import fudge.resonances.resolved as resolvedResonanceModule
from fudge import documentation as documentationModule
from PoPs.chemicalElements.misc import *

import json,sys
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

##############################################  write_Ryaml 
# 
# This is based on ferdinand/write_Ryaml.py but writes the full covariance matrix.
# Adds covIndex data to the Ryaml file.

def write_Ryaml(gnds,outFile,inverse_hessian,border,frontier,GNDS_var,searchloc,norm_info,norm_refs,fixedloc,
                searchnames,searchpars,fixedpars,fixednames,verbose,debug):
  
    print("Write",outFile,'\n')
    ordered = False

    domain = gnds.styles.getEvaluatedStyle().projectileEnergyDomain
    energyUnit = domain.unit
    
    PoPs = gnds.PoPs
    rrr = gnds.resonances.resolved
    Rm_Radius = gnds.resonances.getScatteringRadius()
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs(energyUnit)
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs(energyUnit)

    BC = RMatrix.boundaryCondition
    BV = RMatrix.boundaryConditionValue
    IFG = RMatrix.reducedWidthAmplitudes  
    approximation = RMatrix.approximation
    
    Header = {}
    R_Matrix = {}
    Particles = {}
    Reactions = {}
    SpinGroups = {}
    Data = {}
    Covariances = {}
    
    proj,targ = gnds.projectile,gnds.target
    elasticChannel = '%s + %s' % (proj,targ)
    PoPs = gnds.PoPs    

# HEADER
    Header['projectile'] = proj
    Header['target'] = targ
    Header['evaluation'] = gnds.evaluation
    Header['frame'] = str(gnds.projectileFrame)
    Header['energyUnit'] = energyUnit
    Header['emin'] = emin
    Header['emax'] = emax
    Header['scatteringRadius'] = Rm_global    
    
# R_Matrix
    R_Matrix['approximation'] = str(approximation)
    R_Matrix['reducedWidthAmplitudes'] = IFG
    R_Matrix['boundaryCondition'] = str(BC)
    R_Matrix['boundaryConditionValue'] = str(BV)
    
    
# REACTIONS

    reactionOrder = []
    for pair in RMatrix.resonanceReactions:
        kp = pair.label
        reac = kp.replace(' ','')
        reactionOrder.append(reac)
        
        reaction = pair.link.link
        p,t = pair.ejectile,pair.residual
        projectile = PoPs[p];
        target     = PoPs[t];
        pMass = projectile.getMass('amu');   tMass =     target.getMass('amu');
        if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
        if hasattr(target, 'nucleus'):     target = target.nucleus

        pZ    = projectile.charge[0].value;  tZ  = target.charge[0].value
        pA    = int(pMass+0.5);              tA  = int(tMass+0.5)
        pZA = pZ*1000 + pA;                  tZA = tZ*1000+tA
        cZA = pZA + tZA   # compound nucleus
        if pair.Q is not None:
            QI = pair.Q.getConstantAs(energyUnit)
        else:
            QI = reaction.getQ(energyUnit)
        if pair.getScatteringRadius() is not None:
            prmax =  pair.getScatteringRadius().getValueAs('fm')
        else:
            prmax = Rm_global
        if pair.hardSphereRadius is not None:
            hsrad = pair.hardSphereRadius.getValueAs('fm')
        else:
            hsrad = prmax

        if verbose: print(pMass, tMass, cZA,masses.getMassFromZA( cZA ))
        CN = idFromZAndA(cZA//1000,cZA % 1000)
        
#   proplines = ['Particle & Mass & Charge & Spin & Parity & $E^*$  \\\\ \n','\\hline \n']
        jp,pt,ep = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try:
            jt,tt,et =     target.spin[0].float('hbar'), target.parity[0].value,     target.energy[0].pqu(energyUnit).value
        except:
            jt,tt,et = None,None,None

        Particles[p] = {'gndsName':p, 'gsMass':pMass, 'charge':pZ, 'spin':jp, 'parity':pt, 'excitation':float(ep)}
        Particles[t] = {'gndsName':t, 'gsMass':tMass, 'charge':tZ, 'spin':jt, 'parity':tt, 'excitation':float(et)}

        try:
            CN_PoPs = PoPs[CN]
            CNMass = CN_PoPs.getMass('amu')
            Particles[CN]['gsMass'] = CNMass
            jCN =    CN_PoPs.spin[0].float('hbar')
            Particles[CN]['spin'] = jCN
            pCN =    CN_PoPs.parity[0].value
            Particles[CN]['parity'] = pCN
        except:
            pass
        
        Reactions[reac] = {'label':kp, 'ejectile':p,  'residual':t, 'Q':QI} 
        if prmax != Rm_global:  Reactions[reac]['scatteringRadius'] = prmax
        if hsrad != prmax:      Reactions[reac]['hardSphereRadius'] = hsrad
        
        B = pair.boundaryConditionValue
        if B is not None:       Reactions[reac]['B'] = B

        if pair.label == elasticChannel: 
            lab2cm = tMass / (pMass + tMass)    
            Qelastic = QI

    if debug: print("Elastic channel Q=",Qelastic," with lab2cm factor = %.4f" % lab2cm)
    Reactions['order'] = reactionOrder


### R-MATRIX PARAMETERS
    maxChans = 0
    for Jpi in RMatrix.spinGroups: maxChans = max(maxChans,len(Jpi.channels))
    cols = maxChans + 1

    width_unitsi = 'unknown'
    Overrides = 0
    for Jpi in RMatrix.spinGroups:
        R = Jpi.resonanceParameters.table
        for ch in Jpi.channels:
            if ch.boundaryConditionValue is not None: Overrides += 1

    if BC is None:
        btype = 'S'
    elif BC==resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction:
        btype = 'S'
    elif BC==resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum:
        btype = '-L'
    elif BC==resolvedResonanceModule.BoundaryCondition.Brune:
        btype = 'Brune'
    elif BC==resolvedResonanceModule.BoundaryCondition.Given:
        btype = BV
    else:
        print("Boundary condition BC <%s> not recognized" % BC,"in write_tex")
        raise SystemExit
    if BV is None: BV = ''
        
    if BC != resolvedResonanceModule.BoundaryCondition.Brune: BC = "B = %s" % btype
    boundary = " in the %s basis" %  BC
    if Overrides and verbose: print('  with %s overrides' % Overrides)

    frame = 'lab'
    widthUnit = energyUnit + ('**(1/2)' if IFG==1 else '')
    if debug: print('Boundary conditions are %s : %s in units %s' % (BC,BV,widthUnit))
    
    index = 0
    jset = 0
    spinGroupOrder = []
    for Jpi in RMatrix.spinGroups:
        jtot = str(Jpi.spin)
        parity = int(Jpi.parity)
        pi = '+' if parity>0 else '-'
        if True: print("\nSpin group:",jtot,pi)
        
        spinGroup = jtot + pi
        spinGroupOrder.append(spinGroup)
        group = {}
        
        R = Jpi.resonanceParameters.table
        poleEnergies = R.getColumn('energy',energyUnit)
        widths = [R.getColumn( col.name, widthUnit ) for col in R.columns if col.name != 'energy']
        rows = len(poleEnergies)
        if rows > 0:
            columns = len(R[0][:])
            print(jtot,pi,' ',rows,'poles, each with',columns-1,'widths:',rows*columns,'parameters')
#             nParameters += rows*columns
        else:
            columns = 0
        
        channels = []
        for ch in Jpi.channels:
            n = ch.columnIndex
            rr = ch.resonanceReaction
            rreac = RMatrix.resonanceReactions[rr]
            label = rreac.label
            lch = ch.L
            sch = float(ch.channelSpin)
            B = ch.boundaryConditionValue
            channels.append([str(rr),lch,sch,B])
            
#         print('Pole variables:',GNDS_var[jset,:rows,0])
        poleData = {}
        for i in range(rows):
            tag = 'pole'+str(i).zfill(3)+':'+"%.3f" % R[i][0]
            
            covIndex = index if GNDS_var[jset,i,0]>0 else None
            par = [ [covIndex, float(R[i][0]) ] , [] ]
            if covIndex is not None:  index += 1
            
            for c in range(1,columns):
                covIndex = index if GNDS_var[jset,i,c]>0 else None
                par[1].append( [covIndex, float(R[i][c]) ]  )
                if covIndex is not None:  index += 1
                
            poleData[ tag ] = par
            
        if verbose:
            print('poleData',poleData)
            print('channels',channels)

        group['channels'] = channels
        group['poles']    = poleData
        
        SpinGroups[spinGroup] = group
        jset += 1
    
    numVariables = index
    SpinGroups['order'] = spinGroupOrder
    print('Number of R-matrix parameters:',numVariables) #,nParameters)
    
# DATA NORMS

    normData = {}
    dataOrder = []
    for ip in range(border[2],border[3]):
        name1 = searchnames[ip]
        datanorm = float(searchpars[ip])
        ni = searchloc[ip,0]
        name2,reffile = norm_refs[ni]
        name = name2.replace('r:','')
        expect,chi_scale,fixed = norm_info[ni,:]
        
        dataDict = {}
        dataDict['datanorm'] = datanorm
        dataDict['covIndex'] = ip
        dataDict['filename'] = reffile
        dataDict['shape'] = True if (chi_scale == 0 and fixed == 0) else False
        dataDict['expected'] = float(expect)
        if chi_scale > 0:
            dataDict['syserror'] = 1./float(chi_scale)
            
        normData[name] = dataDict
        dataOrder.append(name)
        if verbose:
            print("Previous norm for %-20s is %10.5f from %s in cov at %s" % (name,datanorm,reffile,ip) )
            
    for ifixed in range(frontier[2],frontier[3]):
        name1 = fixednames[ifixed]
        datanorm = float(fixedpars[ifixed])
        covIndex =  None
        ni = fixedloc[ifixed,0]
        name2,reffile = norm_refs[ni]
        name = name2.replace('r:','')
        expect,chi_scale,fixed = norm_info[ni,:]
                
        dataDict = {}
        dataDict['datanorm'] = datanorm
        dataDict['filename'] = reffile
        dataDict['shape'] =  True if (chi_scale == 0 and fixed == 0) else False
        dataDict['expected'] = float(expect)
        if chi_scale > 0:
            dataDict['syserror'] = 1./float(chi_scale)
        normData[name]  = dataDict
        dataOrder.append(name)
        if verbose:
            print("Fixed norm for %-20s is %10.5f from %s, not in cov" % (name,datanorm,reffile) )        
 
    normData['order'] = dataOrder
    Data['Normalizations']  = normData
            
# COVARIANCES

    Covariances['square matrix'] = inverse_hessian.tolist()

 # Keep Parts order:
 
    Info = {'Header':Header, 'R_Matrix':R_Matrix, 'Particles':Particles, 'Reactions':Reactions, 
            'Spin Groups': SpinGroups, 'Data': Data, 'Covariances':Covariances }
#     output = dump(Info, Dumper=Dumper) 

    Parts = ['Header', 'R_Matrix', 'Particles', 'Reactions', 'Spin Groups', 'Data', 'Covariances']  # ordered
    
    output = ''
    for part in Parts:  # Data.keys():
        d = {part:Info[part]}
        output += dump(d, Dumper=Dumper)

    ofile = open(outFile,'w')
    print(output, file=ofile)
    
    return()
