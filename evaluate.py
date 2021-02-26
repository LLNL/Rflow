
import numpy,os,sys,math
from CoulCF import Pole_Shifts

hbc =   197.3269788e0             # hcross * c (MeV.fm)
finec = 137.035999139e0           # 1/alpha (fine-structure constant)
amu   = 931.4940954e0             # 1 amu/c^2 in MeV

coulcn = hbc/finec                # e^2
fmscal = 2e0 * amu / hbc**2
etacns = coulcn * math.sqrt(fmscal) * 0.5e0
pi = 3.1415926536
rsqr4pi = 1.0/(4*pi)**0.5

# import tensorflow as tf
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

try:
  physical_devices = tf.config.list_physical_devices('GPU')
  print("GPUs:",physical_devices)
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
# print('TF: cannot read and/or modify virtual devices')
  pass
# tf.logging.set_verbosity(tf.logging.ERROR)


def evaluate(Multi,ComputerPrecisions,Channels,CoulombFunctions_data,CoulombFunctions_poles, Dimensions,Logicals, 
                 Search_Control,Data_Control, searchpars0, data_val,tim):
                 
    if Multi > 1:
        strategy = tf.distribute.MirroredStrategy()
        npal = strategy.num_replicas_in_sync
        print('Parallel with',npal)
    elif Multi == 1:
        strategy = tf.distribute.OneDeviceStrategy()
        npal = strategy.num_replicas_in_sync
        print('Parallel with',npal)    
    else:
        strategy = tf.distribute.get_strategy()

#     ComputerPrecisions = (REAL, CMPLX, INT, realSize)
#     Logicals = (LMatrix,brune,Lambda,EBU,chargedElastic, debug,verbose)
#     Dimensions = (n_data,npairs,n_jsets,n_poles,n_chans,n_angles,n_angle_integrals,n_totals,NL,maxpc,batches)
# 
#     Channels = [ipair,nch,npl,pname,tname,za,zb,QI,cm2lab,rmass,prmax,L_val,c0,cn,seg_val]
#     CoulombFunctions_data = (L_diag, Om2_mat,POm_diag,CSp_diag_in,CSp_diag_out, Rutherford, InterferenceAmpl, Gfacc,gfac)    # batch n_data
#     Grid=0:    CoulombFunctions_poles = (S_poles,dSdE_poles,EO_poles,has_widths)                                                  # batch n_jsets
#     Grid>0:    CoulombFunctions_poles = [Lowest_pole_energy,Highest_pole_energy,ShiftE]                  # S on a regular grid

# 
#     Search_Control = (searchloc,border,E_poles_fixed_v,g_poles_fixed_v, fixed_norms,norm_info,effect_norm,data_p, AAL,base,Search,Iterations,widthWeight,restarts,Cross_Sections)
# 
#     Data_Control = [Pleg, ExptAint,ExptTot,CS_diag,p_mask,gfac_s]     # Pleg + extra for Cross-sections  
#
#     searchpars0  # initial values
#  
#     Dataset = data_val                              # batch n_data 

    REAL, CMPLX, INT, realSize = ComputerPrecisions
    LMatrix,brune,Grid,Lambda,EBU,chargedElastic, debug,verbose = Logicals
    n_data,npairs,n_jsets,n_poles,n_chans,n_angles,n_angle_integrals,n_totals,NL,maxpc,batches = Dimensions

    ipair,nch,npl,pname,tname,za,zb,QI,cm2lab,rmass,prmax,L_val,c0,cn,seg_val = Channels
    L_diag, Om2_mat,POm_diag,CSp_diag_in,CSp_diag_out, Rutherford, InterferenceAmpl, Gfacc,gfac = CoulombFunctions_data   # batch n_data
    if Grid == 0.:
        S_poles,dSdE_poles,EO_poles,has_widths = CoulombFunctions_poles
    else:                                               # batch n_jsets
        S_poles,dSdE_poles,EO_poles = CoulombFunctions_poles  # dSdE_poles,EO_pole = Elow,Ehigh (float scalars!)

    LMatrix,brune,Grid,Lambda,EBU,chargedElastic, debug,verbose = Logicals

    searchloc,border,E_poles_fixed_v,g_poles_fixed_v,D_poles_fixed_v, fixed_norms,norm_info,effect_norm,data_p, AAL,base,Search,Iterations,widthWeight,restarts,Cross_Sections = Search_Control

    Pleg, ExptAint,ExptTot,CS_diag,p_mask,gfac_s = Data_Control

#     AAL = numpy.zeros([npairs,npairs, n_jsets,maxpc,maxpc, n_jsets,maxpc,maxpc ,NL], dtype=REAL)

#     AA = numpy.zeros([n_angles, n_jsets,maxpc,maxpc, n_jsets,maxpc,maxpc], dtype=REAL)
    if n_angles > 0:
        AA = []
        for jl in range(n_jsets):
            AA_jl =  numpy.zeros([n_angles, maxpc,maxpc, n_jsets,maxpc,maxpc], dtype=REAL)
            for ie in range(n_angles):
                pin = data_p[ie,0]
                pout= data_p[ie,1]
                for L in range(NL):
                    AA_jl[ie, :,:, :,:,:] += AAL[pout,pin, jl,:,:, :,:,:, L] * Pleg[ie,L]
            AA.append(AA_jl)
    else:
        AA = None

    Tind = numpy.zeros([n_data,n_jsets,maxpc,maxpc,2], dtype=INT) 
    Mind = numpy.zeros([n_data,n_jsets,maxpc,maxpc], dtype=CMPLX)
    print('\nTp_mat size',n_data*n_jsets*(npairs*maxpc)**2*16/1e9,'GB')
    for jset in range(n_jsets):
        for ie in range(n_data):
            pin = data_p[ie,0]
            pout= data_p[ie,1]; 
            if pout == -1: pout = pin # to get total cross-section
            for ci in range(c0[jset,pin],cn[jset,pin]):
                ici = ci - c0[jset,pin]
                for co in range(c0[jset,pout],cn[jset,pout]):
                    ico = co - c0[jset,pout]
#                     if ico>=maxpc: 
#                     print('jset,ie,pin,pout,:,ci,ici,co,ico=',jset,ie,pin,pout,':',ci,ici,co,ico)
                    Tind[ie,jset,ico,ici,:] = numpy.asarray([co,ci])
                    Mind[ie,jset,ico,ici]   = 1.0
#     Tp = tfx.gather_nd(Tmat, Tind, batch_dims=1)
                     
    n_angle_integrals0 = n_angles                # so [n_angle_integrals0,n_totals0] for angle-integrals
    n_totals0 = n_angles + n_angle_integrals     # so [n_totals0:n_data]             for totals
                   
    n_pars = border[4]
    n_norms = fixed_norms.shape[0]
    print('Search parameters :',n_pars)
    n_dof = n_data - n_pars
    print('Data points:',n_data,'of which',n_angles,'are for angles,',n_angle_integrals,'are for angle-integrals, and ',n_totals,'are for totals. Dof=',n_dof)
    sys.stdout.flush()


################################################################    
## TENSORFLOW:
    if Search or (brune and Grid > 0.):
        import tensorflow_probability as tfp

    @tf.function
    def R2T_transformsTF(g_poles,E_rpoles,E_ipoles,E_scat,L_diag, Om2_mat,POm_diag, n_jsets,n_poles,n_chans):
    # Now do TF:
        GL = tf.expand_dims(g_poles,2)
        GR = tf.expand_dims(g_poles,3)

        GG  = GL * GR
        GGe = tf.expand_dims(GG,0)  
        if Lambda is None:
            POLES = tf.reshape(tf.complex(E_rpoles,E_ipoles), [1,n_jsets,n_poles,1,1])  # same for all energies and channel matrix
        else:
            zero = tf.constant(0.0, dtype = REAL)
            if Lambda > 1e-10:
                Dmod = tf.maximum(tf.math.real(E_scat) - EBU ,zero) ** Lambda   # size [n_data]
            elif Lambda < -1e-10:
                Dmod = tf.maximum(tf.sign(tf.math.real(E_scat) - EBU ) - tf.exp( Lambda * (tf.math.real(E_scat) - EBU ) ), zero )
            else:
                Dmod = tf.maximum(tf.sign(tf.math.real(E_scat) - EBU ), zero )  # size [n_data]
            POLES = tf.reshape(tf.complex(E_rpoles, zero), [1,n_jsets,n_poles,1,1])  # real part same for all energies and channel matrix
            POLES +=tf.complex(zero, tf.reshape(Dmod,[-1,1,1,1,1]) * tf.reshape(E_ipoles, [1,n_jsets,n_poles,1,1]) )  # imag part NOT same for all energies
            
        SCAT  = tf.reshape(E_scat,  [-1,1,1,1,1])             # vary only for scattering energies

        RPARTS = GGe / (POLES - SCAT);    # print('RPARTS',RPARTS.dtype,RPARTS.get_shape())

        RMATC = tf.reduce_sum(RPARTS,2)  # sum over poles

        C_mat = tf.eye(n_chans, dtype=CMPLX) - RMATC * tf.expand_dims(L_diag,2);            # print('C_mat',C_mat.dtype,C_mat.get_shape())

        D_mat = tf.linalg.solve(C_mat,RMATC);                                               # print('D_mat',D_mat.dtype,D_mat.get_shape())
    #    S_mat = Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2);
    #  T=I-S
        T_mat = tf.eye(n_chans, dtype=CMPLX) - (Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2) )
                
        return(T_mat)

    @tf.function
    def LM2T_transformsTF(g_poles,E_rpoles,E_ipoles,E_scat,L_diag, Om2_mat,POm_diag, n_jsets,n_poles,n_chans,brune,Grid,S_poles,dSdE_poles,EO_poles):
    # Use Level Matrix A to get T=1-S:
    #     print('g_poles',g_poles.dtype,g_poles.get_shape())

        T_matList = []
        zero = tf.constant(0.0, dtype = REAL)
        
        for js in range(n_jsets):
            m = nch[js]  # number of partial-wave channels
            p = npl[js]  # number of poles
        
            W = tf.reshape(L_diag[:,js,:m],[-1,1,1,m]) #; print('LDIAG',LDIAG.dtype,LDIAG.get_shape())

            if Lambda is None:
#                 POLES = tf.reshape(E_poles[js,:p], [1,p,1])  # same for all energies and channel matrices
                POLES = tf.reshape(tf.complex(E_rpoles[js,:p],E_ipoles[js,:p]), [1,p,1])  # same for all energies and channel matrix
                if brune:
                    POLES_L = tf.reshape(POLES, [1,p,1,1])  # same for all energies and channel matrix
                    POLES_R = tf.reshape(POLES, [1,1,p,1])  # same for all energies and channel matrix
            else:
                if Lambda > 1e-10:
                    Dmod = tf.maximum(tf.math.real(E_scat) - EBU ,zero) ** Lambda   # size [n_data]
                elif Lambda < -1e-10:
                    Dmod = tf.maximum(tf.sign(tf.math.real(E_scat) - EBU ) - tf.exp( Lambda * (tf.math.real(E_scat) - EBU ) ), zero )
                else:
                    Dmod = tf.maximum(tf.sign(tf.math.real(E_scat) - EBU ), zero )
                
                POLES = tf.reshape(tf.complex(E_rpoles[js,:p], zero), [1,p,1])  # real part same for all energies and channel matrix
                POLES +=tf.complex(zero, tf.reshape(Dmod,[-1,1,1]) * tf.reshape(E_ipoles[js,:p], [1,p,1]) )  # imag part NOT same for all energies
                # print(js,p,'POLES',POLES.dtype,POLES.get_shape()) # gives shape [n_data,p,1]
                if brune:
                    POLES_L = tf.reshape(POLES,[-1,p,1,1])
                    POLES_R = tf.reshape(POLES,[-1,1,p,1])

            if brune:   # add extra terms to GLG

                if Grid == 0.0:  # first-order Taylor series
                    SE_poles = S_poles[js,:p,:m] + tf.expand_dims(tf.math.real(E_rpoles[js,:p])-EO_poles[js,:p],1) * dSdE_poles[js,:p,:m]
                else:  # interpolate on regular grid S_poles
                    Elow, Ehigh = dSdE_poles,EO_poles
                    SE_poles = tfp.math.interp_regular_1d_grid(E_rpoles[js,:p], Elow, Ehigh, S_poles[:,js,:m], axis=0)  # want final dimensions SE_poles[:p,:m]
#                     print('\nSE_poles',SE_poles.dtype,SE_poles.get_shape(),'for p,m=',p,m,'\n')
                    
#                 POLES_L = tf.reshape(E_poles[js,:p], [1,p,1,1])  # same for all energies and channel matrix
#                 POLES_R = tf.reshape(E_poles[js,:p], [1,1,p,1])  # same for all energies and channel matrix
                SHIFT_L = tf.reshape(SE_poles[:p,:m], [1,p,1,m] ) # [J,n,c] >  [1,n,1,c]
                SHIFT_R = tf.reshape(SE_poles[:p,:m], [1,1,p,m] ) # [J,n,c] >  [1,1,n,c]
                SCATL  = tf.reshape(E_scat,  [-1,1,1,1])             # vary only for scattering energies

                NUM = tf.complex(SHIFT_L,zero) * (SCATL - POLES_R) - tf.complex(SHIFT_R,zero) * (SCATL - POLES_L)  # expect [ie,n',n,c]
                DEN = POLES_L - POLES_R
                W_offdiag = tf.math.divide_no_nan( NUM , DEN )  
                W_diag    = tf.reshape( tf.eye(p, dtype=CMPLX), [1,p,p,1]) * tf.complex(SHIFT_R,zero) 
                W = W - W_diag - W_offdiag

            GL = tf.reshape(g_poles[js,:p,:m],[1,p,1,m]) #; print('GL',GL.dtype,GL.get_shape())
            GR = tf.reshape(g_poles[js,:p,:m],[1,1,p,m]) #; print('GR',GR.dtype,GR.get_shape())

            GLG = tf.reduce_sum( GL * W * GR , 3)    # giving [ie,n',n]
#             POLES = tf.reshape(E_poles[js,:p], [1,p,1])  # same for all energies and channel matrices
            SCAT  = tf.reshape(E_scat,  [-1,1,1])             # vary only for scattering energies
            Ainv_mat = tf.eye(p, dtype=CMPLX) * (POLES - SCAT) - GLG    # print('Ainv_mat',Ainv_mat.dtype,Ainv_mat.get_shape())

            A_mat = tf.linalg.inv(Ainv_mat);       

            D_mat = tf.matmul( g_poles[js,:p,:], tf.matmul( A_mat, g_poles[js,:p,:]) , transpose_a=True)     # print('D_mat',D_mat.dtype,D_mat.get_shape())

        #    S_mat = Om2_mat + complex(0.,2.) :q:q!* tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2);
        #  T=I-S
            T_matJ =  tf.eye(n_chans, dtype=CMPLX) - (Om2_mat[:,js,:,:] + complex(0.,2.) * tf.expand_dims(POm_diag[:,js,:],2) * D_mat * tf.expand_dims(POm_diag[:,js,:],1) ) 
            T_matList.append( T_matJ ) 
            
        T_mat = tf.stack(T_matList, 1) #; print('T_mat',T_mat.dtype,T_mat.get_shape())

        return(T_mat)

        
    @tf.function
    def T_convertTF(T_mat): #, Tind, Mind):
#         print('T_mat',T_mat.dtype,T_mat.get_shape())
#         print('Tind',Tind.dtype,Tind.shape)
#         print('Mind',Mind.dtype,Mind.shape)

        Tp_mat = tf.gather_nd(T_mat, Tind, batch_dims=2) * tf.constant(Mind)   # ie,jset,p1,c1,p2,c2 in/out partitions for batch data spec.
#         A = tf.gather_nd(T_mat, Tind, batch_dims=2)   # ie,jset,p1,c1,p2,c2 in/out partitions for batch data spec.
#         B =  tf.constant(Mind)   # ie,jset,p1,c1,p2,c2 in/out partitions for batch data spec.
#         print('A',A.dtype,A.get_shape())
#         print('B',B.dtype,B.get_shape())
#         Tp_mat = A * B
        return(Tp_mat)


    @tf.function
    def T2X_transformsTF(Tp_mat,gfac, n_jsets,n_chans,npairs,maxpc):
    
    #for  1 - Re(S) = Re(1-S) = Re(T)
        TOT_mat = tf.math.real(tf.linalg.diag_part(Tp_mat))   #  ie,jset,a,b -> ie,jset,ad   # valid for pin=pout from (ie )
                     
        XS_tot  = TOT_mat * tf.expand_dims(gfac, 2)                          #  ie,jset,a  * gfac[ie,jset,1]
        XSp_tot = 2. *  tf.reduce_sum(  XS_tot, [1,2])     # convert ie,jset,a to ie by summing over jset,a


        Tmod2 = tf.math.real(  Tp_mat * tf.math.conj(Tp_mat) )   # ie,jset,ao,ai
        XSp_mat = tf.reduce_sum (Tmod2 * tf.reshape(gfac, [-1,n_jsets,1,1] ), [1,2,3])  # sum over jset,ao,ai  giving ie which implies po,pi
                        
        return(XSp_mat,XSp_tot) 

    @tf.function
    def T2B_transformsTF(TCp_mat,AA, n_jsets,n_chans,n_angles):

    #  T= T_mat[:,n_jsets,npairs,maxpc,npairs,maxpc]
        T_left = tf.reshape(TCp_mat[:n_angles,:,:,:],  [-1,n_jsets,maxpc,maxpc, 1,1,1])  #; print(' T_left', T_left.get_shape())
        T_right= tf.reshape(TCp_mat[:n_angles,:,:,:],  [-1,1,1,1, n_jsets,maxpc,maxpc])  #; print(' T_right', T_right.get_shape())

        if n_angles > 0:
            Ax = tf.zeros( AA[0].shape[0], dtype=REAL)
            for jl in range(n_jsets):
                TAT = AA[jl][:,:,:, :,:,:] * tf.math.real( tf.math.conj(T_left[:, jl,:,:, :,:,:]) * T_right[:, 0,:,:, :,:,:] )
                Ax += tf.reduce_sum(TAT,[ 1,2, 3,4,5])    # exlude dim=0 (ie)
        else:
            Ax = tf.zeros(0, dtype=REAL)
            
        return(Ax)  
                            
    @tf.function
    def AddCoulombsTF(A_t,  Rutherford, InterferenceAmpl, TCp_mat_pdiag, Gfacc, n_angles):
    
        return(( A_t + Rutherford + tf.reduce_sum (tf.math.imag( InterferenceAmpl * tf.linalg.diag_part(TCp_mat_pdiag[:n_angles,:,:,:]) ) , [1,2])) * Gfacc )

    @tf.function
    def ChiSqTF(A_t, widthWeight,searchWidths, data_val,norm_val,norm_info,effect_norm):

    # chi from cross-sections
        one = tf.constant(1.0, dtype=REAL)
        fac = tf.reduce_sum(tf.expand_dims(norm_val[:]-one,0) * effect_norm, 1) + one

        chi = (A_t/fac/data_val[:,4] - data_val[:,2])/data_val[:,3]
        chisq = tf.reduce_sum(chi**2)

    # chi from norm_vals themselves:
        chi = (norm_val - norm_info[:,0]) * norm_info[:,1]
        chisq += tf.reduce_sum(chi**2)

        chisq += tf.reduce_sum(searchWidths**4) * widthWeight

        return (chisq)

    @tf.function
    def FitMeasureTF(searchpars):

        E_pole_v = tf.scatter_nd (searchloc[border[0]:border[1],:] ,   searchpars[border[0]:border[1]], [n_jsets*n_poles] )
        g_pole_v = tf.scatter_nd (searchloc[border[1]:border[2],:] ,   searchpars[border[1]:border[2]], [n_jsets*n_poles*n_chans] )
        norm_valv= tf.scatter_nd (searchloc[border[2]:border[3],:] ,   searchpars[border[2]:border[3]], [n_norms] )
        D_pole_v = tf.scatter_nd (searchloc[border[3]:border[4],:] ,   searchpars[border[3]:border[4]], [n_jsets*n_poles] )

        E_rpoles =         tf.reshape(E_pole_v+E_poles_fixed_v,[n_jsets,n_poles])
        E_ipoles =  -0.5 * tf.reshape(D_pole_v+D_poles_fixed_v,[n_jsets,n_poles])**2 
        g_cpoles = tf.complex(tf.reshape(g_pole_v+g_poles_fixed_v,[n_jsets,n_poles,n_chans]),tf.constant(0., dtype=REAL))
        E_cscat  = tf.complex(data_val[:,0],tf.constant(0., dtype=REAL)) 
        norm_val =                       (norm_valv+ fixed_norms)**2

        if not LMatrix:
             T_mat =  R2T_transformsTF(g_cpoles,E_rpoles,E_ipoles,E_cscat,L_diag, Om2_mat,POm_diag, n_jsets,n_poles,n_chans ) 
        else:
             T_mat = LM2T_transformsTF(g_cpoles,E_rpoles,E_ipoles,E_cscat,L_diag, Om2_mat,POm_diag, n_jsets,n_poles,n_chans,brune,Grid,S_poles,dSdE_poles,EO_poles ) 

        Tp_mat = T_convertTF(T_mat) #, Tind, Mind)

    # multiply left and right by Coulomb phases:
        TCp_mat = tf.expand_dims(CSp_diag_out,3) * Tp_mat[:n_angles,...] * tf.expand_dims(CSp_diag_in,2)
    
        Ax = T2B_transformsTF(TCp_mat,AA, n_jsets,n_chans,n_angles)

        if chargedElastic:                          
            AxA = AddCoulombsTF(Ax,  Rutherford, InterferenceAmpl, TCp_mat[:,:,:,:], Gfacc, n_angles)
        else:
            AxA =  Ax * Gfacc
    
        XSp_mat,XSp_tot  = T2X_transformsTF(Tp_mat,gfac, n_jsets,n_chans,npairs,maxpc)

        AxI = XSp_mat[n_angle_integrals0:n_totals0] 
        AxT = XSp_tot[n_totals0:n_data] 
    
        A_t = tf.concat([AxA, AxI, AxT], 0)
        chisq = ChiSqTF(A_t, widthWeight,searchpars[border[1]:border[2]], data_val,norm_val,norm_info,effect_norm)
        ww = tf.reduce_sum(searchpars[border[1]:border[2]]**4) * widthWeight
        
        if Search:
            tf.print(chisq/n_data, ww/n_data, (chisq-ww)/n_data,                             output_stream=trace)
            tf.print(chisq/n_data, ww/n_data, (chisq-ww)/n_data, searchpars,  summarize=-1,   output_stream=snap)
    
        return(chisq, tf.gradients(chisq, searchpars)[0] )
    

    with strategy.scope():
        
        searchpars = tf.Variable(searchpars0)
            
        if Search:
            os.system("rm -f %s/%s-bfgs_min.trace" % (base,base) ) 
            os.system("rm -f %s/%s-bfgs_min.snap" % (base,base) )
            trace = "file://%s/%s-bfgs_min.trace" % (base,base)
            snap = "file://%s/%s-bfgs_min.snap"  % (base,base) 
                        
        print("First FitStatusTF: ",tim.toString( ))
        
        
        if Multi > 0:
            dataset1a= tf.data.Dataset.from_tensor_slices ( (InterferenceAmpl,CSp_diag_in,CSp_diag_out))
            dataset2a= tf.data.Dataset.from_tensor_slices ( (Rutherford, Gfacc))  
            dataset3a = tf.data.Dataset.from_tensor_slices ( [ AA[jl] for jl in range(n_jsets) ] )
            print('dataset1a:\n',dataset1a)
            print('dataset2a:\n',dataset2a)
            print('dataset3a:\n',dataset3a)
    #         dataset_23a = dataset2a.concatenate(dataset3a)
    #         print('dataset_23a:\n',dataset_23a)
        
            dataset1t= tf.data.Dataset.from_tensor_slices ( (data_val,effect_norm,gfac) )
            dataset2t= tf.data.Dataset.from_tensor_slices ( (POm_diag,L_diag))
            dataset3t= tf.data.Dataset.from_tensor_slices ( Om2_mat )
            print('dataset1t:\n',dataset1t)
            print('dataset2t:\n',dataset2t)
            print('dataset3t:\n',dataset3t)
    #         dataset_23t = dataset2t.concatenate(dataset3t)
    #         print('dataset_23t:\n',dataset_23t)
        
                
            dataset_t = dataset1t.concatenate(dataset2t)
            print('dataset5:\n',dataset5)

            print('dataset0:\n',dataset0)

# dataset1a:
#  <TensorSliceDataset shapes: ((10, 3), (10, 3), (10, 3)), types: (tf.complex128, tf.complex128, tf.complex128)>
# dataset2a:
#  <TensorSliceDataset shapes: ((), ()), types: (tf.float64, tf.float64)>
# dataset3a:
#  <TensorSliceDataset shapes: (1809, 3, 3, 10, 3, 3), types: tf.float64>
# dataset1t:
#  <TensorSliceDataset shapes: ((5,), (11,), (10,)), types: (tf.float64, tf.float64, tf.float64)>
# dataset2t:
#  <TensorSliceDataset shapes: ((10, 4), (10, 4)), types: (tf.complex128, tf.complex128)>
# dataset3t:
#  <TensorSliceDataset shapes: (10, 4, 4), types: tf.complex128>
# 
#     dataset_t = dataset1t.concatenate(dataset2t)
# TypeError: Two datasets to concatenate have different types (tf.float64, tf.float64, tf.float64) and (tf.complex128, tf.complex128)
        
            def split_data_model(n):
                AAn = [ AA[jl][n::npal,...]  for jl in range(n_jsets)]
                datas = (data_val[n::npal,:],effect_norm[n::npal,:],gfac[n::npal,:],
                        Rutherford[n::npal],InterferenceAmpl[n::npal,:,:],Gfacc[n::npal], AAn, 
                        CSp_diag_in[n::npal,:,:],CSp_diag_out[n::npal,:,:],Om2_mat[n::npal,:,:,:],POm_diag[n::npal,:,:],
                        L_diag[n::npal,:,:]  )
                model = (n,npal, n_angles_n,n_angle_integrals_n,n_data_n, n_jsets,n_poles,n_chans,maxpc,npairs,nch,npl,
                         LMatrix,brune,chargedElastic,
                         searchloc,border,E_poles_fixed_v,g_poles_fixed_v,fixed_norms,norm_info,
                         S_poles,dSdE_poles,EO_poles )
                this_part = (datas,model)
                return this_part
    #         dist_dataset = my_strategy.experimental_distribute_dataset(dataset)   HOW FOR MY DATA??
            
            n_angles_n = stride_size(0,n_angles, npal)
            n_angle_integrals_n = stride_size(n_angles,n_angles+n_angle_integrals ,npal)
            n_data_n = stride_size(0,n_data, npal)
        
            tot_chisq, tot_Grads = 0.0, tf.zeros_like(searchpars)
            for n in range(npal):
                this_part = split_data_model(n)
        
                per_replica_chisq = strategy.run( FitMeasureTF, args=(searchpars,this_part[0],this_part[1]) )
    #             tf.print('per_replica_chisq:',per_replica_chisq)
    #             tf.print('per_replica_chisq[1]:',per_replica_chisq[1])
                chisq_n = strategy.reduce( tf.distribute.ReduceOp.SUM,  per_replica_chisq[0], axis = None)
                Grads_n = strategy.reduce( tf.distribute.ReduceOp.SUM,  per_replica_chisq[1], axis = None)
    #             tf.print('Grads_n:',Grads_n)
                tot_chisq += chisq_n
                tot_Grads += Grads_n 
            chisq0,Grads = tot_chisq, tot_Grads 
    #         tf.print('Grads:',Grads)
   

                
        chisq0,Grads = FitMeasureTF(searchpars)  # returning values in global   A_tF,Grads,  Tp_mat, XSp_mat,XSp_tot              

        chisq0_n = chisq0.numpy()
        grad0 = Grads.numpy()
        
        ww = ( tf.reduce_sum(searchpars[border[1]:border[2]]**4) * widthWeight ).numpy()
        print('\nFirst run chisq/pt:',chisq0_n/n_data,'including ww/pt',ww/n_data,':',(chisq0_n-ww)/n_data ,'for data.\n') 
         
        if verbose: print('Grads:',grad0)
        sys.stdout.flush()

        if Search:
    
            import tensorflow_probability as tfp
            print('Start search'); sys.stdout.flush()
            optim_results = tfp.optimizer.bfgs_minimize (FitMeasureTF, initial_position=searchpars,
                                max_iterations=Iterations, tolerance=float(Search))
                            
            last_cppt = optim_results.objective_value.numpy()/n_data
            searchpars = optim_results.position
            
            for restart in range(restarts):
                searchpars_n = searchpars.numpy()

                print('More pole energies:',searchpars_n[border[0]:border[1]])
                print('Before restart',restart,' objective chisq/pt',last_cppt)
                print('And objective FitMeasureTF =',FitMeasureTF(searchpars)[0].numpy()/n_data )
            
                if brune and Grid == 0.0:
                    E_pole_v = tf.scatter_nd (searchloc[border[0]:border[1],:] ,   searchpars[border[0]:border[1]], [n_jsets*n_poles] )
                    g_pole_v = tf.scatter_nd (searchloc[border[1]:border[2],:] ,   searchpars[border[1]:border[2]], [n_jsets*n_poles*n_chans] )
                    norm_valv= tf.scatter_nd (searchloc[border[2]:border[3],:] ,   searchpars[border[2]:border[3]], [n_norms] )
                    D_pole_v = tf.scatter_nd (searchloc[border[3]:border[4],:] ,   searchpars[border[3]:border[4]], [n_jsets*n_poles] )

                    E_rpoles =         tf.reshape(E_pole_v+E_poles_fixed_v,[n_jsets,n_poles])
                    E_ipoles =  -0.5 * tf.reshape(D_pole_v+D_poles_fixed_v,[n_jsets,n_poles])**2
                    g_cpoles = tf.complex(tf.reshape(g_pole_v+g_poles_fixed_v,[n_jsets,n_poles,n_chans]),tf.constant(0., dtype=REAL))
                    E_cscat  = tf.complex(data_val[:,0],tf.constant(0., dtype=REAL)) 
                    norm_val =                       (norm_valv+ fixed_norms)**2

                    EOO_poles = EO_poles.copy()
                    SOO_poles = S_poles.copy()
                    for ip in range(border[0],border[1]): #### Extract parameters after previous search:
                        i = searchloc[ip,0]
                        jset = i//n_poles;  n = i%n_poles
                        EO_poles[jset,n] = searchpars_n[ip]
                        
                    # Re-evaluate pole shifts
                    Pole_Shifts(S_poles,dSdE_poles, EO_poles,has_widths, seg_val,1./cm2lab[ipair],QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
                
                    if debug:                         # Print out differences in shifts:
                        for jset in range(n_jsets):
                            for n in range(n_poles):
                                print('j/n=',jset,n,' E old,new:',EOO_poles[jset,n],EO_poles[jset,n])
                                for c in range(n_chans):
                                     print('      S old,new %10.6f, %10.6f, expected %5.2f %%' % (SOO_poles[jset,n,c],S_poles[jset,n,c],
                                             100*dSdE_poles[jset,n,c]*(EO_poles[jset,n]-EOO_poles[jset,n])/ (S_poles[jset,n,c] - SOO_poles[jset,n,c])))
                    
                    Tp_mat = LM2T_transformsTF(g_cpoles,E_rpoles,E_ipoles,E_cscat,L_diag, Om2_mat,POm_diag, n_jsets,n_poles,n_chans,brune,None,S_poles,dSdE_poles,EO_poles ) 

                    XSp_mat,XSp_tot  = T2X_transformsTF(Tp_mat,gfac, n_jsets,n_chans,npairs,maxpc)
        
                # multiply left and right by Coulomb phases:
                    TCp_mat = tf.expand_dims(CSp_diag_out,3) * Tp_mat[:n_angles,...] * tf.expand_dims(CSp_diag_in,2)
                            
                    Ax  = T2B_transformsTF(TCp_mat,AA, n_jsets,n_chans,n_angles)
                    AxA = AddCoulombsTF(Ax,  Rutherford, InterferenceAmpl, TCp_mat[:,:,:,:], Gfacc, n_angles)
                    
                    AxI = XSp_mat[n_angle_integrals0:n_totals0] 
                    AxT = XSp_tot[n_totals0:n_data] 
                    
                    A_t = tf.concat([AxA, AxI, AxT], 0) 
    
                    chisq = ChiSqTF(A_t, widthWeight,searchpars[border[1]:border[2]], data_val,norm_val,norm_info,effect_norm)
                    print('  After pole shifts: ChiSqTF =',chisq.numpy()/n_data )
                    sys.stdout.flush()
                
                optim_results = tfp.optimizer.bfgs_minimize (FitMeasureTF, initial_position=searchpars,
                        max_iterations=Iterations, tolerance=float(Search))
                searchpars = optim_results.position
                new_cppt = optim_results.objective_value.numpy()/n_data
#               if new_cppt >= last_cppt: break
                last_cppt = new_cppt
                      
            print('\nResults:')
            print('Converged:',optim_results.converged.numpy(), 'Failed:',optim_results.failed.numpy())
            print('Num_iterations:',optim_results.num_iterations.numpy(), 'Num_objective_evaluations:',optim_results.num_objective_evaluations.numpy())
            
            chisqF = optim_results.objective_value
            chisqF_n = chisqF.numpy()
            print('Objective_value:',chisqF_n, 'Objective chisq/pt',chisqF_n/n_data)
            
            searchpars = optim_results.position
            searchpars_n = searchpars.numpy()
#             print('position:\n',searchpars_n)
            
            inverse_hessian = optim_results.inverse_hessian_estimate.numpy()
            print('inverse_hessian shape=',inverse_hessian.shape ,'\ndiagonal:',['%6.4f,' % inverse_hessian[i,i] for i in range(n_pars)] )
#             print(dir(optim_results))
        
        else:
            inverse_hessian = None
            searchpars_n = searchpars0
        
        print("Second FitStatusTF start: ",tim.toString( ))
        
        chisqF,Grads = FitMeasureTF(searchpars)  # A_tF,Grads,  Tp_mat, XSp_mat,XSp_tot via globals
        grad1 = Grads.numpy()
        chisqF_n = chisqF.numpy()
        chisqpptF_n = chisqF_n/n_data
        chisqpdofF_n = chisqF_n/n_dof
        print(  'chisq from FitStatusTF/pt:',chisqpptF_n,' chisq/dof =',chisqpdofF_n)
        if verbose: print('Grads:',grad1)
#  END OF STRATEGY

    if Cross_Sections:
        strategy = tf.distribute.get_strategy()
    
        Tind = None; Mind = None
        TAind = numpy.zeros([n_data,n_jsets,npairs,maxpc,npairs,maxpc,2], dtype=INT) 
        MAind = numpy.zeros([n_data,n_jsets,npairs,maxpc,npairs,maxpc], dtype=CMPLX)  # mask: 1 = physically valid
        
    
        print('TAp_mat size',n_data*n_jsets*(npairs*maxpc)**2*16/1e9,'GB')
        for jset in range(n_jsets):
            for np1 in range(npairs):
                for np2 in range(npairs):
                     for c1 in range(c0[jset,np1],cn[jset,np1]):
                         ic1 = c1 - c0[jset,np1]
                         for c2 in range(c0[jset,np2],cn[jset,np2]):
                             ic2 = c2 - c0[jset,np2]
                             TAind[:,jset,np1,ic1,np2,ic2,:] = numpy.asarray([c1,c2]) 
                             MAind[:,jset,np1,ic1,np2,ic2]   = 1.0 
                         
                         
        TCind = numpy.zeros([n_angles,n_jsets,maxpc,maxpc,2], dtype=INT) 
        MCind = numpy.zeros([n_angles,n_jsets,maxpc,maxpc], dtype=CMPLX)
        print('TCp_mat size',n_angles*n_jsets*(npairs*maxpc)**2*16/1e9,'GB')
        for jset in range(n_jsets):
            for ie in range(n_angles):
                pin = data_p[ie,0]
                pout= data_p[ie,1]
                for ci in range(c0[jset,pin],cn[jset,pin]):
                    ici = ci - c0[jset,pin]
                    for co in range(c0[jset,pout],cn[jset,pout]):
                        ico = co - c0[jset,pout]
                        TCind[ie,jset,ico,ici,:] = numpy.asarray([co,ci])
                        MCind[ie,jset,ico,ici]   = 1.0

        @tf.function
        def T_convert_s(T_mat):
            # multiply left and right by Coulomb phases:
            TC_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)

            TAp_mat = tf.gather_nd(T_mat, TAind, batch_dims=2) * tf.constant(MAind)
            if n_angles>0:
                TCp_mat = tf.gather_nd(TC_mat[:n_angles,...], TCind, batch_dims=2) * tf.constant(MCind)  # ie,jset,p1,c1,p2,c2
            else:
                TCp_mat = tf.zeros_like(MCind)
                
            return( TAp_mat,TCp_mat )
            
        @tf.function
        def T2X_transforms_s(TAp_mat,CS_diag,gfac_s,p_mask, n_jsets,n_chans,npairs,maxpc):
    
            nm = npairs*maxpc
            TOT_mat = tf.reshape( 
                           tf.math.real(tf.linalg.diag_part(tf.reshape(TAp_mat,[-1,n_jsets,nm,nm])) )
                         ,[-1,n_jsets,npairs,maxpc])   #  ie,jset,pd,ad  for  1 - Re(S) = Re(1-S) = Re(T)
                     
            XS_tot  = TOT_mat * gfac_s                          #  ie,jset,p,a 
            XSp_tot = 2. *  tf.reduce_sum(  XS_tot, [1,3])     # convert ie,jset,p,a to ie,p by summing over jset,a


            Tmod2 = tf.math.real(  TAp_mat * tf.math.conj(TAp_mat) )   # ie,jset,po,ao,pi,ai
            XSp_mat = tf.reduce_sum (Tmod2 * tf.reshape(gfac_s, [-1,n_jsets,1,1,npairs,maxpc] ), [1,3,5])  # sum over jset,ao,ai  giving ie,po,pi
                        
            XSp_cap = XSp_tot - tf.reduce_sum(XSp_mat,1)  # total - sum of xsecs(pout)

            return(XSp_mat,XSp_tot,XSp_cap) 

        with strategy.scope():

            E_pole_v = tf.scatter_nd (searchloc[border[0]:border[1],:] ,      searchpars[border[0]:border[1]], [n_jsets*n_poles] )
            g_pole_v = tf.scatter_nd (searchloc[border[1]:border[2],:] ,      searchpars[border[1]:border[2]], [n_jsets*n_poles*n_chans] )
            norm_valv= tf.scatter_nd (searchloc[border[2]:border[3],:] ,   searchpars[border[2]:border[3]], [n_norms] )
            D_pole_v = tf.scatter_nd (searchloc[border[3]:border[4],:] ,   searchpars[border[3]:border[4]], [n_jsets*n_poles] )

            E_rpoles =         tf.reshape(E_pole_v+E_poles_fixed_v,[n_jsets,n_poles])
            E_ipoles =  -0.5 * tf.reshape(D_pole_v+D_poles_fixed_v,[n_jsets,n_poles])**2
            g_cpoles = tf.complex(tf.reshape(g_pole_v + g_poles_fixed_v,[n_jsets,n_poles,n_chans]),tf.constant(0., dtype=REAL))
            E_cscat  = tf.complex(data_val[:,0],tf.constant(0., dtype=REAL)) 
            norm_val =                       (norm_valv+ fixed_norms)**2

            if not LMatrix:
                 T_mat =  R2T_transformsTF(g_cpoles,E_rpoles,E_ipoles,E_cscat,L_diag, Om2_mat,POm_diag, n_jsets,n_poles,n_chans ) 
            else:
                 T_mat = LM2T_transformsTF(g_cpoles,E_rpoles,E_ipoles,E_cscat,L_diag, Om2_mat,POm_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles) 
                 
            TAp_mat,TCp_mat = T_convert_s(T_mat)
    
            Ax = T2B_transformsTF(TCp_mat,AA, n_jsets,n_chans,n_angles)

            if chargedElastic:                          
                AxA = AddCoulombsTF(Ax,  Rutherford, InterferenceAmpl, TCp_mat[:,:,:,:], Gfacc, n_angles)
            else:
                AxA = Ax * Gfacc
            
            XSp_mat,XSp_tot,XSp_cap  = T2X_transforms_s(TAp_mat,CS_diag,gfac_s,p_mask, n_jsets,n_chans,npairs,maxpc)
            
            AxI = tf.reduce_sum(XSp_mat[n_angle_integrals0:n_totals0,:,:] * ExptAint, [1,2])   # sum over pout,pin
            AxT = tf.reduce_sum(XSp_tot[n_totals0:n_data,:] * ExptTot, 1)   # sum over pin

            A_tF = tf.concat([AxA,AxI,AxT], 0)
            chisq = ChiSqTF(A_tF, widthWeight,searchpars[border[1]:border[2]], data_val,norm_val,norm_info,effect_norm)

            print("First FitStatusTF: ",tim.toString( ),'giving chisq/pt',chisq.numpy()/n_data)
    
            A_tF_n = A_tF.numpy()
            XS_totals = [XSp_tot.numpy(),XSp_cap.numpy(), XSp_mat.numpy()]
    
    else:
        A_tF_n, XS_totals = None, None
        
#  END OF TENSORFLLOW
    print("Ending tf: ",tim.toString( ))
###################################################

    return( searchpars_n, chisqF_n, grad1, inverse_hessian,  chisq0_n,grad0, A_tF_n, XS_totals)

