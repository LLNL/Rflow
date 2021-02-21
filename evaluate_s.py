
import numpy,os,sys,math
from CoulCF import Pole_Shifts


hbc =   197.3269788e0             # hcross * c (MeV.fm)
finec = 137.035999139e0           # 1/alpha (fine-structure constant)
amu   = 931.4940954e0             # 1 amu/c^2 in MeV

coulcn = hbc/finec                # e^2
fmscal = 2e0 * amu / hbc**2
etacns = coulcn * math.sqrt(fmscal) * 0.5e0
pi = 3.1415926536

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

# strategy = tf.distribute.MirroredStrategy()

# based on evaluates4.py

def evaluate_s(ComputerPrecisions,Channels,CoulombFunctions_data,CoulombFunctions_poles, Dimensions,Logicals, 
                 Search_Control,Data_Control, searchpars0, data_val,tim):


#     ComputerPrecisions = (REAL, CMPLX, INT, realSize)
# 
#     Channels = [ipair,nch,npl,pname,tname,za,zb,QI,cm2lab,rmass,prmax,L_val,c0,cn,seg_val]
#     CoulombFunctions_data = (L_diag, Om2_mat,POm_diag,Rutherford, InterferenceAmpl, Gfacc,gfac)    # batch n_data
#     CoulombFunctions_poles = (S_poles,dSdE_poles,EO_poles,has_widths)                                                  # batch n_jsets
# 
#     Dimensions = (n_data,npairs,n_jsets,n_poles,n_chans,n_angles,n_angle_integrals,n_totals,NL,maxpc,batches)
#     Logicals = (LMatrix,brune,chargedElastic, debug,verbose)
# 
#     Search_Control = (searchloc,border,E_poles_fixed_v,g_poles_fixed_v, fixed_norms,norm_info,effect_norm,data_p, AAL,base,Search,Iterations,widthWeight,restarts)
# 
#     Data_Control = (Pleg, ExptAint,ExptTot,CS_diag,p_mask,gfac_s)     # batch n_angle_integrals,  n_totals  
                                                #   (Pleg, AAL) or AA
#     searchpars0  # initial values
#  
#     Dataset = data_val                              # batch n_data 
#     tim

    REAL, CMPLX, INT, realSize = ComputerPrecisions

    ipair,nch,npl,pname,tname,za,zb,QI,cm2lab,rmass,prmax,L_val,c0,cn,seg_val = Channels
    L_diag, Om2_mat,POm_diag,CSp_diag_in,CSp_diag_out, Rutherford, InterferenceAmpl, Gfacc,gfac = CoulombFunctions_data   # batch n_data
    S_poles,dSdE_poles,EO_poles,has_widths = CoulombFunctions_poles                                                  # batch n_jsets

    n_data,npairs,n_jsets,n_poles,n_chans,n_angles,n_angle_integrals,n_totals,NL,maxpc,batches = Dimensions
    LMatrix,brune,chargedElastic, debug,verbose = Logicals

    searchloc,border,E_poles_fixed_v,g_poles_fixed_v, fixed_norms,norm_info,effect_norm,data_p, AAL,base,Search,Iterations,widthWeight,restarts = Search_Control

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
                         
    Tind = numpy.zeros([n_angles,n_jsets,maxpc,maxpc,2], dtype=INT) 
    Mind = numpy.zeros([n_angles,n_jsets,maxpc,maxpc], dtype=CMPLX)
    print('TCp_mat size',n_angles*n_jsets*(npairs*maxpc)**2*16/1e9,'GB')
    for jset in range(n_jsets):
        for ie in range(n_angles):
            pin = data_p[ie,0]
            pout= data_p[ie,1]
            for ci in range(c0[jset,pin],cn[jset,pin]):
                ici = ci - c0[jset,pin]
                for co in range(c0[jset,pout],cn[jset,pout]):
                    ico = co - c0[jset,pout]
                    Tind[ie,jset,ico,ici,:] = numpy.asarray([co,ci])
                    Mind[ie,jset,ico,ici]   = 1.0
                     
    n_angle_integrals0 = n_angles                # so [n_angle_integrals0,n_totals0] for angle-integrals
    n_totals0 = n_angles + n_angle_integrals     # so [n_totals0:n_data]             for totals
                   
    n_pars = border[3]
    n_norms = fixed_norms.shape[0]
    print('Search parameters :',n_pars)
    ndof = n_data - n_pars
    print('Data points:',n_data,'of which',n_angles,'are for angles,',n_angle_integrals,'are for angle-integrals, and ',n_totals,'are for totals. Dof=',ndof)
    sys.stdout.flush()


################################################################    
## TENSORFLOW:

#   with strategy.scope():
    if True:
        
        searchpars = tf.Variable(searchpars0)
            
 
        @tf.function
        def R2T_transforms_s(g_poles,E_poles,E_scat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans):
        # Now do TF:
            GL = tf.expand_dims(g_poles,2)
            GR = tf.expand_dims(g_poles,3)

            GG  = GL * GR
            GGe = tf.expand_dims(GG,0)  

            POLES = tf.reshape(E_poles, [1,n_jsets,n_poles,1,1])  # same for all energies and channel matrix
            SCAT  = tf.reshape(E_scat,  [-1,1,1,1,1])             # vary only for scattering energies

            RPARTS = GGe / (POLES - SCAT);    # print('RPARTS',RPARTS.dtype,RPARTS.get_shape())

            RMATC = tf.reduce_sum(RPARTS,2)  # sum over poles

            C_mat = tf.eye(n_chans, dtype=CMPLX) - RMATC * tf.expand_dims(L_diag,2);            # print('C_mat',C_mat.dtype,C_mat.get_shape())

            D_mat = tf.linalg.solve(C_mat,RMATC);                                               # print('D_mat',D_mat.dtype,D_mat.get_shape())

        #    S_mat = Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2);
        #  T=I-S
            T_mat = tf.eye(n_chans, dtype=CMPLX) - (Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2) )
    
        # multiply left and right by Coulomb phases:
            TC_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)
            
            TAp_mat = tf.gather_nd(T_mat, TAind, batch_dims=2) * MAind  #  all in/out partitions. No Coulomb phases/
            if n_angles>0:
                TCp_mat = tf.gather_nd(TC_mat[:n_angles,...], Tind, batch_dims=2) * Mind   #  in/out partitions for batch data spec. With Coulomb phases.
            else:
                TCp_mat = tf.zeros_like(Mind)
            
            return( TAp_mat,TCp_mat)

        @tf.function
        def LM2T_transforms_s(g_poles,E_poles,E_scat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles):
        # Use Level Matrix A to get T=1-S:
        #     print('g_poles',g_poles.dtype,g_poles.get_shape())

            T_matList = []
            for js in range(n_jsets):
                m = nch[js]  # number of partial-wave channels
                p = npl[js]  # number of poles
            
                W = tf.reshape(L_diag[:,js,:m],[-1,1,1,m]) #; print('LDIAG',LDIAG.dtype,LDIAG.get_shape())

                if brune:   # add extra terms to GLG
                    Z = tf.constant(0.0, dtype=REAL)
                    SE_poles = S_poles[js,:p,:m] + tf.expand_dims(tf.math.real(E_poles[js,:p])-EO_poles[js,:p],1) * dSdE_poles[js,:p,:m]
                    POLES_L = tf.reshape(E_poles[js,:p], [1,p,1,1])  # same for all energies and channel matrix
                    POLES_R = tf.reshape(E_poles[js,:p], [1,1,p,1])  # same for all energies and channel matrix
                    SHIFT_L = tf.reshape(SE_poles[:p,:m], [1,p,1,m] ) # [J,n,c] >  [1,n,1,c]
                    SHIFT_R = tf.reshape(SE_poles[:p,:m], [1,1,p,m] ) # [J,n,c] >  [1,1,n,c]
                    SCATL  = tf.reshape(E_scat,  [-1,1,1,1])             # vary only for scattering energies

                    NUM = tf.complex(SHIFT_L,Z) * (SCATL - POLES_R) - tf.complex(SHIFT_R,Z) * (SCATL - POLES_L)  # expect [ie,n',n,c]
                    DEN = POLES_L - POLES_R
                    W_offdiag = tf.math.divide_no_nan( NUM , DEN )  
                    W_diag    = tf.reshape( tf.eye(p, dtype=CMPLX), [1,p,p,1]) * tf.complex(SHIFT_R,Z) 
                    W = W - W_diag - W_offdiag

                GL = tf.reshape(g_poles[js,:p,:m],[1,p,1,m]) #; print('GL',GL.dtype,GL.get_shape())
                GR = tf.reshape(g_poles[js,:p,:m],[1,1,p,m]) #; print('GR',GR.dtype,GR.get_shape())

                GLG = tf.reduce_sum( GL * W * GR , 3)    # giving [ie,n',n]
                POLES = tf.reshape(E_poles[js,:p], [1,p,1])  # same for all energies and channel matrix
                SCAT  = tf.reshape(E_scat,  [-1,1,1])             # vary only for scattering energies
                Ainv_mat = tf.eye(p, dtype=CMPLX) * (POLES - SCAT) - GLG    # print('Ainv_mat',Ainv_mat.dtype,Ainv_mat.get_shape())
    
                A_mat = tf.linalg.inv(Ainv_mat);       
    
                D_mat = tf.matmul( g_poles[js,:p,:], tf.matmul( A_mat, g_poles[js,:p,:]) , transpose_a=True)     # print('D_mat',D_mat.dtype,D_mat.get_shape())

            #    S_mat = Om2_mat + complex(0.,2.) :q:q!* tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2);
            #  T=I-S
                T_matJ =  tf.eye(n_chans, dtype=CMPLX) - (Om2_mat[:,js,:,:] + complex(0.,2.) * tf.expand_dims(POm_diag[:,js,:],2) * D_mat * tf.expand_dims(POm_diag[:,js,:],1) ) 
                T_matList.append( T_matJ ) 
                
            T_mat = tf.stack(T_matList, 1) #; print('T_mat',T_mat.dtype,T_mat.get_shape())
        # multiply left and right by Coulomb phases:
            TC_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)

            TAp_mat = tf.gather_nd(T_mat, TAind, batch_dims=2) * tf.constant(MAind)
            if n_angles>0:
                TCp_mat = tf.gather_nd(TC_mat[:n_angles,...], Tind, batch_dims=2) * tf.constant(Mind)  # ie,jset,p1,c1,p2,c2
            else:
                TCp_mat = tf.zeros_like(Mind)
                
            return( TAp_mat,TCp_mat)
            
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

        @tf.function
        def T2B_transforms_s(TCp_mat,AA, n_jsets,n_chans,n_angles,batches):

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
        def AddCoulombs_s(A_t,  Rutherford, InterferenceAmpl, TCp_mat_pdiag, Gfacc, n_angles):
        
            return(( A_t + Rutherford + tf.reduce_sum (tf.math.imag( InterferenceAmpl * tf.linalg.diag_part(TCp_mat_pdiag[:n_angles,:,:,:]) ) , [1,2])) * Gfacc )

        @tf.function
        def ChiSq_s(A_t, widthWeight,searchWidths, data_val,norm_val,norm_info,effect_norm):
    
        # chi from cross-sections
            one = tf.constant(1.0, dtype=REAL)
            fac = tf.reduce_sum(tf.expand_dims(norm_val[:]-one,0) * effect_norm, 1) + one

            chi = (A_t/fac/data_val[:,4] - data_val[:,2])/data_val[:,3]
            chisq = tf.reduce_sum(chi**2)

        # chi from norm_vals themselves:
            chi = (norm_val - norm_info[:,0]) * norm_info[:,1]
            chisq += tf.reduce_sum(chi**2)
            
            chisq += tf.reduce_sum(searchWidths**2) * widthWeight
    
            return (chisq)
                        
        @tf.function
        def FitStatus_s(searchpars):

            E_pole_v = tf.scatter_nd (searchloc[border[0]:border[1],:] ,      searchpars[border[0]:border[1]], [n_jsets*n_poles] )
            g_pole_v = tf.scatter_nd (searchloc[border[1]:border[2],:] ,      searchpars[border[1]:border[2]], [n_jsets*n_poles*n_chans] )
            norm_valv= tf.scatter_nd (searchloc[border[2]:border[3],:] ,      searchpars[border[2]:border[3]], [n_norms] )
    
            E_cpoles = tf.complex(tf.reshape(E_pole_v + E_poles_fixed_v,[n_jsets,n_poles]),        tf.constant(0., dtype=REAL)) 
            g_cpoles = tf.complex(tf.reshape(g_pole_v + g_poles_fixed_v,[n_jsets,n_poles,n_chans]),tf.constant(0., dtype=REAL))
            E_cscat  = tf.complex(data_val[:,0],tf.constant(0., dtype=REAL)) 
            norm_val =                       (norm_valv+ fixed_norms)**2
    
            if not LMatrix:
                 TAp_mat,TCp_mat = R2T_transforms_s(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans ) 
            else:
                 TAp_mat,TCp_mat = LM2T_transforms_s(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles) 
        

            Ax = T2B_transforms_s(TCp_mat,AA, n_jsets,n_chans,n_angles,batches)

            if chargedElastic:                          
                AxA = AddCoulombs_s(Ax,  Rutherford, InterferenceAmpl, TCp_mat[:,:,:,:], Gfacc, n_angles)
            else:
                AxA = Ax * Gfacc
                
            XSp_mat,XSp_tot,XSp_cap  = T2X_transforms_s(TAp_mat,CS_diag,gfac_s,p_mask, n_jsets,n_chans,npairs,maxpc)
                
            AxI = tf.reduce_sum(XSp_mat[n_angle_integrals0:n_totals0,:,:] * ExptAint, [1,2])   # sum over pout,pin
            AxT = tf.reduce_sum(XSp_tot[n_totals0:n_data,:] * ExptTot, 1)   # sum over pin

            A_t = tf.concat([AxA,AxI,AxT], 0)
            chisq = ChiSq_s(A_t, widthWeight,searchpars[border[1]:border[2]], data_val,norm_val,norm_info,effect_norm)

#           Grads = tf.gradients(chisq, searchpars) 
            Grads = [tf.zeros(n_pars) ]

            return(chisq,A_t,Grads,  TAp_mat,TCp_mat, XSp_mat,XSp_tot,XSp_cap)
        
        print("First FitStatusTF: ",tim.toString( ))

        chisq0,A_tF,Grads,  TAp_mat,TCp_mat, XSp_mat,XSp_tot,XSp_cap = FitStatus_s(searchpars)                 
                            
        A_tF_n = A_tF.numpy()
        chisq0_n = chisq0.numpy()
        ww = ( tf.reduce_sum(searchpars[border[1]:border[2]]**2) * widthWeight ).numpy()

        print('\nFirst run:',chisq0_n/n_data,'including ww',ww/n_data,':',(chisq0_n-ww)/n_data ,'for data.\n') 

        grad0 = Grads[0].numpy()
        if verbose: print('Grads:',grad0)

        if debug:
            TAp_mat_n = TAp_mat.numpy()
            SMAT = numpy.zeros(n_chans, dtype=CMPLX)
            for ie in range(n_data):
                for jset in range(n_jsets):
                    print('Energy',data_val[ie,0],' jset=',jset ) #  J=',J_set[jset],pi_set[jset],'\n S-matrix is size',seg_col[jset])
                    for a in range(n_chans):
                        for b in range(n_chans):
                            npa = seg_val[jset,a] 
                            npb = seg_val[jset,b]
                            ca = a - c0[jset,npa]
                            cb = b - c0[jset,npb]
                            SMAT[b] = (1 if a==b else 0) - numpy.conj(CS_diag[ie,jset,a]) * TAp_mat_n[ie,jset,npa,ca,npb,cb]  # remove Coulomb phases
                        print('   ',a,'row: ',',  '.join(['{:.5f}'.format(SMAT[b]) for b in range(n_chans)]) )                    
#                         print('   ',a,'row: ',',  '.join(['{:.5f}'.format(T_mat[ie,jset,a,b].numpy()) for b in range(n_chans)]) )
    
        if verbose:
            if n_angles>0: xsFile = open(base + '/' + base + '.xsa','w')
            Angular_XS = A_tF.numpy()
            chisqsum = 0.0      
            for ie in range(n_angles):
                fac = 1.0
                for ni in range(n_norms):
                    fac += (norm_val[ni]-1.)*effect_norm[ni,ie]
                chi = (Angular_XS[ie]/fac/data_val[ie,4]-data_val[ie,2])/data_val[ie,3] 
                chisqsum += chi**2
                theta = math.acos(mu_val[ie])*180./pi if mu_val[ie] <= 1.0 else -1.
        
                print('%8.2f %8.2f %10.3f %10.3f     %6.1f %6.2f' % (data_val[ie,0],theta,Angular_XS[ie]/data_val[ie,4],data_val[ie,2]*fac,chi,fac), file=xsFile)

    # chi from norm_vals themselves:
            print('\nchisq/pt=',chisqsum/(n_data),'before constrained norms' )
            chisqnorms = 0.
            for ni in range(n_norms):
                chi = (norm_val[ni] - norm_info[ni,0]) * norm_info[ni,1]        
                chisqnorms += chi**2
     
            chisqsum += chisqnorms
            print('chisq/pt=',chisqsum/(n_data),'(including)' )
        
#         inverse_hessian = None
#         searchpars_n = searchpars0
#         
#         print("Second FitStatusTF start: ",tim.toString( ))
# #         chisqF,A_tF,Grads,  TAp_mat,TCp_mat, XSp_mat,XSp_tot,XSp_cap = FitStatus_s(searchpars) 
#         chisqF_n = chisqF.numpy()
#         A_tF_n = A_tF.numpy()
#         grad1 = Grads[0].numpy()
#         print(  'chisq from FitStatusTF:',chisqF_n)
        
        XS_totals = [XSp_tot.numpy(),XSp_cap.numpy(), XSp_mat.numpy()]

#  END OF TENSORFLOW
###################################################
    print("Ending tf: ",tim.toString( ))

    return( chisq0_n/n_data, A_tF_n, XS_totals )
