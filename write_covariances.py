
##############################################
#                                            #
#    Rflow 0.20      Ian Thompson, LLNL      #
#                                            #
# SPDX-License-Identifier: MIT               #
##############################################

import numpy

import fudge.covariances.covarianceSuite as covarianceSuiteModule
import fudge.covariances.covarianceSection as covarianceSectionModule
import fudge.covariances.modelParameters as covarianceModelParametersModule
import fudge.covariances.enums as covarianceEnumsModule

import xData.xDataArray as arrayModule


# Copy covariance matrix back into GNDS 
def write_gnds_covariances(gnds,searchpars,inverse_hessian,GNDS_loc,POLE_details,searchnames,border, base, verbose,debug):
                                   
    nVaried = border[2]  # number of varied parameters (energies, widths, but not norms)

    if True:  # print out covariance matrix for E and widths (not yet norms and D!)
        cov_file =  open ("%s/bfgs_min.cov"  % base,'w')
        print(nVaried,'square covariance matrix for GNDS variables in run',base, file=cov_file)
        for i in range(nVaried):
            det = POLE_details[i]

#   ('E',jset,n,ip,float(J_set[jset]),int(pi_set[jset]))   i=ip, so not printed again
            if det[0] == 'E':
                print(i,'E',det[1],det[2],det[4],det[5],'=','j,n,ip, J,pi:',searchpars[i],searchnames[i], file=cov_file)

#   ('W',jset,n,ip,seg_val[jset,c],L_val[jset,c],S_val[jset,c])   i=ip, so not printed again
            if det[0] == 'W':
                print(i,'W',det[1],det[2],det[4],det[5],det[6],'j,n,ip, RLS:',searchpars[i],searchnames[i], file=cov_file)
        for i in range(nVaried):
            print(i, ' '.join(['%10s' % inverse_hessian[i,j] for j in range(nVaried)]), file=cov_file)
        print('Covariance metadata and matrix written in file',"%s/bfgs_min.cov"  % base,'\n')


 # store into GNDS (need links to each spinGroup)
    parameters = covarianceModelParametersModule.Parameters()
    startIndex = 0
    for spinGroup in gnds.resonances.resolved.evaluated:
        nParams = spinGroup.resonanceParameters.table.nColumns * spinGroup.resonanceParameters.table.nRows
        if nParams == 0: continue
        parameters.add( covarianceModelParametersModule.ParameterLink(
            label = spinGroup.label, link = spinGroup.resonanceParameters.table, root="$reactions",
            matrixStartIndex=startIndex, nParameters=nParams
        ))
        startIndex += nParams
    nRvariables = startIndex

    matrix = numpy.zeros([nRvariables,nRvariables])
    for i in range(nVaried):
        for j in range(nVaried):
            matrix[GNDS_loc[i,0],GNDS_loc[j,0]] = inverse_hessian[i,j]
    
    if debug: 
        print(type(matrix))
        print('matrix:\n',matrix)
        
    if verbose: # given eigenvalues
        correlation = numpy.zeros([nRvariables,nRvariables])
        if debug: print("Cov shape",matrix.shape,", Corr shape",correlation.shape)
        # print "\nCov diagonals:",[matrix[i,i] for i in range(nRvariables)]
        # print "\nCov diagonals:\n",numpy.array_repr(numpy.diagonal(matrix),max_line_width=100,precision=3)
        print("Diagonal uncertainties:\n",numpy.array_repr(numpy.sqrt(numpy.diagonal(matrix)),max_line_width=100,precision=3))
        for i in range(nRvariables):
            for j in range(nRvariables): 
                t = matrix[i,i]*matrix[j,j]
                if t !=0: correlation[i,j] = matrix[i,j] / t**0.5

        from scipy.linalg import eigh
        eigval,evec = eigh(matrix)
        if debug:
            print("  Covariance eigenvalue     Vector")
            for kk in range(nRvariables):
                k = nRvariables-kk - 1
                print(k,"%11.3e " % eigval[k] , numpy.array_repr(evec[:,k],max_line_width=200,precision=3, suppress_small=True))
        else:
            print("Covariance eivenvalues:\n",numpy.array_repr(numpy.flip(eigval[:]),max_line_width=100,precision=3))


        if True: print('correlation matrix:\n',correlation)
        eigval,evec = eigh(correlation)
        if debug:
            print("  Correlation eigenvalue     Vector")
            for kk in range(nRvariables):
                k = nRvariables-kk - 1
                print(k,"%11.3e " % eigval[k] , numpy.array_repr(evec[:,k],max_line_width=200,precision=3, suppress_small=True))
        else:
            print("Correlation eivenvalues:\n",numpy.array_repr(numpy.flip(eigval[:]),max_line_width=100,precision=3))

    GNDSmatrix = arrayModule.Flattened.fromNumpyArray(matrix, symmetry=arrayModule.Symmetry.lower)
    # print GNDSmatrix.toXML()
    Type=covarianceEnumsModule.Type.absolute
    covmatrix = covarianceModelParametersModule.ParameterCovarianceMatrix('eval', GNDSmatrix,
        parameters, type=Type )
    if verbose: print(covmatrix.toXML())
    rowData = covarianceSectionModule.RowData(gnds.resonances.resolved.evaluated,
            root='')
    parameterSection = covarianceModelParametersModule.ParameterCovariance("resolved resonances", rowData)
    parameterSection.add(covmatrix)

    covarianceSuite = covarianceSuiteModule.CovarianceSuite(  gnds.projectile, gnds.target, 'Rflow R-matrix covariances', interaction="nuclear")
    covarianceSuite.parameterCovariances.add(parameterSection)
 
    evalStyle = gnds.styles.getEvaluatedStyle().copy()
    covarianceSuite.styles.add( evalStyle )

#   if debug: print(covarianceSuite.toXMLList())
    if debug: 
        print('Write covariances to CovariancesSuite.xml')
        covarianceSuite.saveToFile('CovariancesSuite.xml')
#     if verbose: 
#   covarianceSuite.saveToFile('CovariancesSuite.xml')
    
    return covarianceSuite
