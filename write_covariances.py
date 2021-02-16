
import numpy

import fudge.gnds.covariances.covarianceSuite as covarianceSuiteModule
import fudge.gnds.covariances.section as covarianceSectionModule
import fudge.gnds.covariances.modelParameters as covarianceModelParametersModule

import xData.xDataArray as arrayModule


# Copy covariance matrix back into GNDS 
def write_gnds_covariances(gnds,inverse_hessian,GNDS_loc,border,  verbose,debug):
                                   
    nVaried = border[2]  # number of varied parameters (energies, widths, but not norms)

 # store into GNDS (need links to each spinGroup)
    parameters = covarianceModelParametersModule.parameters()
    startIndex = 0
    for spinGroup in gnds.resonances.resolved.evaluated:
        nParams = spinGroup.resonanceParameters.table.nColumns * spinGroup.resonanceParameters.table.nRows
        if nParams == 0: continue
        parameters.add( covarianceModelParametersModule.parameterLink(
            label = spinGroup.label, link = spinGroup.resonanceParameters.table, root="$reactions",
            matrixStartIndex=startIndex, nParameters=nParams
        ))
        startIndex += nParams
    nRvariables = startIndex

    matrix = numpy.zeros([nRvariables,nRvariables])
    for i in range(nVaried):
        for j in range(nVaried):
            matrix[GNDS_loc[i],GNDS_loc[j]] = inverse_hessian[i,j]
    
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

    GNDSmatrix = arrayModule.flattened.fromNumpyArray(matrix, symmetry=arrayModule.symmetryLowerToken)
    # print GNDSmatrix.toXML()
    Type="absoluteCovariance"
    covmatrix = covarianceModelParametersModule.parameterCovarianceMatrix('eval', GNDSmatrix,
        parameters, type=Type )
    if verbose: print(covmatrix.toXML())
    rowData = covarianceSectionModule.rowData(gnds.resonances.resolved.evaluated,
            root='')
    parameterSection = covarianceModelParametersModule.parameterCovariance("resolved resonances", rowData)
    parameterSection.add(covmatrix)

    covarianceSuite = covarianceSuiteModule.covarianceSuite(  gnds.projectile, gnds.target, 'Rflow R-matrix covariances' )
    covarianceSuite.parameterCovariances.add(parameterSection)

    if debug: print(covarianceSuite.toXMLList())
#     if verbose: 
#   covarianceSuite.saveToFile('CovariancesSuite.xml')
    
    return covarianceSuite
