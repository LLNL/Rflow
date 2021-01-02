
import numpy,os
from printExcitationFunctions import *

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

strategy = tf.distribute.MirroredStrategy()

def evaluate_tf(ComputerPrecisions,Channels,CoulombFunctions_data,CoulombFunctions_poles, Dimensions,Logicals, 
                 Search_Control,Data_Control, searchpars0, data_val,tim):


#     ComputerPrecisions = (REAL, CMPLX, INT)
# 
#     Channels = [ipair,pname,tname,za,zb,QI,cm2lab,rmass,prmax,L_val]
#     CoulombFunctions_data = (L_diag, Om2_mat,POm_diag,CS_diag, Rutherford, InterferenceAmpl, Gfacc,gfac)    # batch n_data
#     CoulombFunctions_poles = (S_poles,dSdE_poles,EO_poles)                                                  # batch n_jsets
# 
#     Dimensions = (n_data,npairs,n_jsets,n_poles,n_chans,n_angles,n_angle_integrals,n_totals,NL,batches)
#     Logicals = (LMatrix,brune,chargedElastic, TransitionMatrix,debug,verbose)
# 
#     Search_Control = (searchloc,border,E_poles_fixed_v,g_poles_fixed_v, norm_info,effect_norm,p_mask,data_p, AAL,base,Search,Iterations,restarts)
# 
#     Data_Control = (Pleg, ExptAint,ExptTot)     # batch n_angle_integrals,  n_totals  
                                                #   (Pleg, AAL) or AA
#     searchpars0  # initial values
#  
#     Dataset = data_val                              # batch n_data 
#     tim

    REAL, CMPLX, INT = ComputerPrecisions

    ipair,pname,tname,za,zb,QI,cm2lab,rmass,prmax,L_val = Channels
    L_diag, Om2_mat,POm_diag,CS_diag, Rutherford, InterferenceAmpl, Gfacc,gfac = CoulombFunctions_data   # batch n_data
    S_poles,dSdE_poles,EO_poles = CoulombFunctions_poles                                                  # batch n_jsets

    n_data,npairs,n_jsets,n_poles,n_chans,n_angles,n_angle_integrals,n_totals,NL,batches = Dimensions
    LMatrix,brune,chargedElastic, TransitionMatrix,debug,verbose = Logicals

    searchloc,border,E_poles_fixed_v,g_poles_fixed_v, norm_info,effect_norm,p_mask,data_p, AAL,base,Search,Iterations,restarts = Search_Control

    Pleg, ExptAint,ExptTot = Data_Control


    AA = numpy.zeros([n_angles, n_jsets,n_chans,n_chans, n_jsets,n_chans,n_chans], dtype=REAL)
    for ie in range(n_angles):
        pin = data_p[ie,0]
        pout= data_p[ie,1]
        for L in range(NL):
            AA[ie, :,:,:, :,:,:] += AAL[pin,pout, :,:,:, :,:,:, L] * Pleg[ie,L]

    n_angle_integrals0 = n_angles                # so [n_angle_integrals0,n_totals0] for angle-integrals
    n_totals0 = n_angles + n_angle_integrals     # so [n_totals0:n_data]             for totals
                   
    n_pars = border[2]
    print('Search parameters :',n_pars)
    ndof = n_data - n_pars
    print('Data points:',n_data,'of which',n_angles,'are for angles,',n_angle_integrals,'are for angle-integrals, and ',n_totals,'are for totals. Dof=',ndof)


################################################################    
## TENSORFLOW:

#   with strategy.scope():
    if True:
        
        searchpars = tf.Variable(searchpars0)
            
 
        @tf.function
        def R2T_transformsTF(g_poles,E_poles,E_scat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans):
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
            T_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)
    
            return(T_mat)

        @tf.function
        def LM2T_transformsTF(g_poles,E_poles,E_scat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles):
        # Use Level Matrix A to get T=1-S:
        #     print('g_poles',g_poles.dtype,g_poles.get_shape())
            GL = tf.reshape(g_poles,[1,n_jsets,n_poles,1,n_chans]) #; print('GL',GL.dtype,GL.get_shape())
            GR = tf.reshape(g_poles,[1,n_jsets,1,n_poles,n_chans]) #; print('GR',GR.dtype,GR.get_shape())
            LDIAG = tf.reshape(L_diag,[-1,n_jsets,1,1,n_chans]) #; print('LDIAG',LDIAG.dtype,LDIAG.get_shape())
            GLG = tf.reduce_sum( GL * LDIAG * GR , 4)    # giving [ie,J,n',ncd Rf]
            Z = tf.constant(0.0, dtype=REAL)
            if brune:   # add extra terms to GLG
                SE_poles = S_poles + tf.expand_dims(tf.math.real(E_poles)-EO_poles,2) * dSdE_poles
                POLES_L = tf.reshape(E_poles, [1,n_jsets,n_poles,1,1])  # same for all energies and channel matrix
                POLES_R = tf.reshape(E_poles, [1,n_jsets,1,n_poles,1])  # same for all energies and channel matrix
                SHIFT_L = tf.reshape(SE_poles, [1,n_jsets,n_poles,1,n_chans] ) # [J,n,c] >  [1,J,n,1,c]
                SHIFT_R = tf.reshape(SE_poles, [1,n_jsets,1,n_poles,n_chans] ) # [J,n,c] >  [1,J,1,n,c]
                SCAT  = tf.reshape(E_scat,  [-1,1,1,1,1])             # vary only for scattering energies
        #         NUM = SHIFT_L * (SCAT - POLES_R) - SHIFT_R * (SCAT - POLES_L)  # expect [ie,J,n',n,c]
                NUM = tf.complex(SHIFT_L,Z) * (SCAT - POLES_R) - tf.complex(SHIFT_R,Z) * (SCAT - POLES_L)  # expect [ie,J,n',n,c]
                DEN = POLES_L - POLES_R
                W_offdiag = tf.math.divide_no_nan( NUM , DEN )  
                W_diag    = tf.reshape( tf.eye(n_poles, dtype=CMPLX), [1,1,n_poles,n_poles,1]) * tf.complex(SHIFT_R,Z) 
                W = W_diag + W_offdiag
                GLG = GLG - tf.reduce_sum( GL * W * GR , 4)

            POLES = tf.reshape(E_poles, [1,n_jsets,n_poles,1])  # same for all energies and channel matrix
            SCAT  = tf.reshape(E_scat,  [-1,1,1,1])             # vary only for scattering energies
            Ainv_mat = tf.eye(n_poles, dtype=CMPLX) * (POLES - SCAT) - GLG    # print('Ainv_mat',Ainv_mat.dtype,Ainv_mat.get_shape())
    
            A_mat = tf.linalg.inv(Ainv_mat);       
    
            D_mat = tf.matmul( g_poles, tf.matmul( A_mat, g_poles) , transpose_a=True)     # print('D_mat',D_mat.dtype,D_mat.get_shape())

        #    S_mat = Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2);
        #  T=I-S
            T_mat = tf.eye(n_chans, dtype=CMPLX) - (Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2) )
    
        # multiply left and right by Coulomb phases:
            T_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)

            return(T_mat)
            
        @tf.function
        def T2X_transformsTF(T_mat,CS_diag,gfac,p_mask, n_jsets,n_chans,npairs):
                    
            Tmod2 = tf.math.real(  T_mat * tf.math.conj(T_mat) )   # ie,jset,a1,a2

        # sum of Jpi sets:
            G_fac = tf.reshape(gfac, [-1,n_jsets,1,n_chans])
            XS_mat = Tmod2 * G_fac                          # ie,jset,a1,a2   
    
            TOT_mat = tf.math.real(tf.linalg.diag_part(T_mat)* tf.math.conj(CS_diag)**2)   #  ie,jset,a  for  1 - Re(S) = Re(1-S) = Re(T), removing Coulomb phases for TOT
            XS_tot  = TOT_mat * gfac                           #  ie,jset,a
            p_mask1_in = tf.reshape(p_mask, [-1,npairs,n_jsets,n_chans] )   # convert pair,jset,a to  ie,pair,jset,a
            XSp_tot = 2. *  tf.reduce_sum( tf.expand_dims(XS_tot,1) * p_mask1_in , [2,3])     # convert ie,pair,jset,a to ie,pair by summing over jset,a

        
            p_mask_in = tf.reshape(p_mask,[1,1,npairs,n_jsets,1,n_chans])   ;# print('p_mask_in',p_mask_in.get_shape())   # 1,1,pin,jset,1,cin
            p_mask_out =tf.reshape(p_mask,[1,npairs,1,n_jsets,n_chans,1])   ;# print('p_mask_out',p_mask_out.get_shape()) # 1,pout,1,jset,cout,1
    
            XS_ext  = tf.reshape(XS_mat, [-1,1,1,n_jsets,n_chans,n_chans] ) ;# print('XS_ext',XS_ext.get_shape())
            XS_cpio =  XS_ext * p_mask_in * p_mask_out                      ;# print('XS_cpio',XS_cpio.get_shape())
            XSp_mat  = tf.reduce_sum(XS_cpio,[-3,-2,-1] )               # sum over jset,cout,cin, leaving ie,pout,pin
                            
            XSp_cap = XSp_tot - tf.reduce_sum(XSp_mat,1)  # total - sum of xsecs(pout)

            return(XSp_mat,XSp_tot,XSp_cap) 

        
        @tf.function
        def T2B_transformsTF(T_mat,AA, n_jsets,n_chans,n_angles,batches):

        # BB[ie,L] = sum(i,j) T[ie,i]* AA[i,L,j] T[ie,j]
        #  T= T_mat[:,n_jsets,n_chans,n_chans]

            # print(' AA', AA.get_shape())
            T_left = tf.reshape(T_mat[:n_angles,:,:],  [-1,n_jsets,n_chans,n_chans, 1,1,1])  #; print(' T_left', T_left.get_shape())
            T_right= tf.reshape(T_mat[:n_angles,:,:],  [-1,1,1,1, n_jsets,n_chans,n_chans])  #; print(' T_right', T_right.get_shape())
    
            TAT = AA * tf.math.real( tf.math.conj(T_left) * T_right )
        #     TAT = AA * ( tf.math.real(T_left) * tf.math.real(T_right) + tf.math.imag(T_left) * tf.math.imag(T_right) )

            Ax = tf.reduce_sum(TAT,[ 1,2,3, 4,5,6])    # exlude dim=0 (ie)
            return(Ax)
    
                    
        @tf.function
        def AddCoulombsTF(A_t,  Rutherford, InterferenceAmpl, T_mat, Gfacc, n_angles):
        
            return(( A_t + Rutherford + tf.reduce_sum (tf.math.imag( InterferenceAmpl * tf.linalg.diag_part(T_mat[:n_angles,:,:]) ) , [1,2])) * Gfacc )

        @tf.function
        def ChiSqTF(A_t, data_val,norm_val,norm_info,effect_norm):
    
        # chi from cross-sections
            one = tf.constant(1.0, dtype=REAL)
            fac = tf.reduce_sum(tf.expand_dims(norm_val[:]-one,1) * effect_norm, 0) + one

            chi = (A_t/fac/data_val[:,4] - data_val[:,2])/data_val[:,3]
            chisq = tf.reduce_sum(chi**2)

        # chi from norm_vals themselves:
            chi = (norm_val - norm_info[:,0]) * norm_info[:,1]
            chisq += tf.reduce_sum(chi**2)
    
            return (chisq)
                        
        @tf.function
        def FitStatusTF(searchpars):

            E_pole_v = tf.scatter_nd (searchloc[:border[0],:] ,          searchpars[:border[0]],          [n_jsets*n_poles] )
            g_pole_v = tf.scatter_nd (searchloc[border[0]:border[1],:] , searchpars[border[0]:border[1]], [n_jsets*n_poles*n_chans] )
            norm_val = tf.exp( searchpars[border[1]:border[2]] )
    
            E_cpoles = tf.complex(tf.reshape(E_pole_v + E_poles_fixed_v,[n_jsets,n_poles]),        tf.constant(0., dtype=REAL)) 
            g_cpoles = tf.complex(tf.reshape(g_pole_v + g_poles_fixed_v,[n_jsets,n_poles,n_chans]),tf.constant(0., dtype=REAL))
            E_cscat  = tf.complex(data_val[:,0],tf.constant(0., dtype=REAL)) 
    
            if not LMatrix:
                T_mat = R2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans ) 
            else:
                T_mat = LM2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles) 
        

            Ax = T2B_transformsTF(T_mat,AA[:, :,:,:, :,:,:], n_jsets,n_chans,n_angles,batches)

            if chargedElastic:                          
                AxA = AddCoulombsTF(Ax,  Rutherford, InterferenceAmpl, T_mat, Gfacc, n_angles)
            else:
                AxA = Ax * Gfacc
                
            XSp_mat,XSp_tot,XSp_cap  = T2X_transformsTF(T_mat,CS_diag,gfac,p_mask, n_jsets,n_chans,npairs)
                
            AxI = tf.reduce_sum(XSp_mat[n_angle_integrals0:n_totals0,:,:] * ExptAint, [1,2])   # sum over pout,pin
            AxT = tf.reduce_sum(XSp_tot[n_totals0:n_data,:] * ExptTot, 1)   # sum over pin

            A_t = tf.concat([AxA,AxI,AxT], 0)
            chisq = ChiSqTF(A_t, data_val,norm_val,norm_info,effect_norm)

            Grads = tf.gradients(chisq, searchpars) 

            return(chisq,A_t,Grads, T_mat,XSp_mat,XSp_tot,XSp_cap)
        
        print("First FitStatusTF: ",tim.toString( ))

        chisq0,A_tF,Grads, T_mat,XSp_mat,XSp_tot,XSp_cap = FitStatusTF(searchpars)                 
        A_tF_n = A_tF.numpy()
        chisq0_n = chisq0.numpy()

        print('\nFirst run:',chisq0_n/n_data)  
        print("End FitStatusTF: ",tim.toString( ))

#       chisq,A_tF,Grads, T_mat,XSp_mat,XSp_tot,XSp_cap = FitStatusTF(searchpars)                 
#       print("Second tf: ",tim.toString( ))

        grad0 = Grads[0].numpy()
        if verbose: print('Grads:',grad0)
 ###################################################

        if debug:
            T_mat_n = T_mat.numpy()
            SMAT = numpy.zeros(n_chans, dtype=CMPLX)
            for ie in range(n_data):
                for jset in range(n_jsets):
                    print('Energy',E_scat[ie],'  J=',J_set[jset],pi_set[jset],'\n S-matrix is size',seg_col[jset])
                    for a in range(n_chans):
                        for b in range(n_chans):
                            SMAT[b] = (1 if a==b else 0) - numpy.conj(CS_diag[ie,jset,a]) * T_mat_n[ie,jset,a,b] * numpy.conj(CS_diag[ie,jset,a])  # remove Coulomb phases
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
 ###################################################

        if Search:
            os.system("rm -f %s/%s-bfgs_min.trace" % (base,base) ) 
            os.system("rm -f %s/%s-bfgs_min.snap" % (base,base) )
            trace = "file://%s/%s-bfgs_min.trace" % (base,base)
            snap = "file://%s/%s-bfgs_min.snap"  % (base,base) 
            n_pars = border[2]
            ndof = n_data - n_pars
        
            @tf.function        
            def FitMeasureTF(searchpars):

                E_pole_v = tf.scatter_nd (searchloc[:border[0],:] ,          searchpars[:border[0]],          [n_jsets*n_poles] )
                g_pole_v = tf.scatter_nd (searchloc[border[0]:border[1],:] , searchpars[border[0]:border[1]], [n_jsets*n_poles*n_chans] )
                norm_val = tf.exp( searchpars[border[1]:border[2]] )
    
                E_cpoles = tf.complex(tf.reshape(E_pole_v+E_poles_fixed_v,[n_jsets,n_poles]),        tf.constant(0., dtype=REAL)) 
                g_cpoles = tf.complex(tf.reshape(g_pole_v+g_poles_fixed_v,[n_jsets,n_poles,n_chans]),tf.constant(0., dtype=REAL))
                E_cscat  = tf.complex(data_val[:,0],tf.constant(0., dtype=REAL)) 

                if not LMatrix:
                    T_mat = R2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans ) 
                else:
                    T_mat = LM2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles ) 

                Ax = T2B_transformsTF(T_mat,AA[:, :,:,:, :,:,:], n_jsets,n_chans,n_angles,batches)

                if chargedElastic:                          
                    AxA = AddCoulombsTF(Ax,  Rutherford, InterferenceAmpl, T_mat, Gfacc, n_angles)
                else:
                    AxA =  Ax * Gfacc
            
                XSp_mat,XSp_tot,XSp_cap  = T2X_transformsTF(T_mat,CS_diag,gfac,p_mask, n_jsets,n_chans,npairs)
    
                AxI = tf.reduce_sum(XSp_mat[n_angle_integrals0:n_totals0,:,:] * ExptAint, [1,2])   # sum over pout,pin
                AxT = tf.reduce_sum(XSp_tot[n_totals0:n_data,:] * ExptTot, 1)   # sum over pin
            
                A_t = tf.concat([AxA, AxI, AxT], 0)
                chisq = ChiSqTF(A_t, data_val,norm_val,norm_info,effect_norm)
            
                tf.print(chisq/n_data,         output_stream=trace)
                tf.print(chisq/n_data, searchpars,  summarize=-1,   output_stream=snap)
            
                return(chisq, tf.gradients(chisq, searchpars)[0] )
    
            initial_objective = FitMeasureTF(searchpars) 
            chisq0 = initial_objective[0]
            grad0 = initial_objective[1].numpy()
            chisq0_n = chisq0.numpy()
            print('Initial position:',chisq0_n/n_data )
            if verbose: print('Initial grad:',grad0 )
    
            import tensorflow_probability as tfp   
            optim_results = tfp.optimizer.bfgs_minimize (FitMeasureTF, initial_position=searchpars,
                                max_iterations=Iterations, tolerance=float(Search))
                            
            last_cppt = optim_results.objective_value.numpy()/n_data
            searchpars = optim_results.position
            
            for restart in range(restarts):
                searchpars_n = optim_results.position.numpy()

                print('More pole energies:',searchpars_n[:border[0]])
                print('Before restart',restart,' objective chisq/pt',last_cppt)
                print('And objective FitMeasureTF =',FitMeasureTF(searchpars)[0].numpy()/n_data )
            
                if brune:
                    EOO_poles = EO_poles.copy()
                    SOO_poles = S_poles.copy()
                    for ip in range(border[0]): #### Extract parameters after previous search:
                        i = searchloc[ip,0]
                        jset = i//n_poles;  n = i%n_poles
                        EO_poles[jset,n] = searchpars[ip]
                        
                    # Re-evaluate pole shifts
                    Pole_Shifts(S_poles,dSdE_poles, EO_poles,has_widths, seg_val,1./cm2lab[ipair],QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
                
                    # Print out differences in shifts:
                    for jset in range(n_jsets):
                        for n in range(n_poles):
                            print('j/n=',jset,n,' E old,new:',EOO_poles[jset,n],EO_poles[jset,n])
                            for c in range(n_chans):
                                 print('      S old,new %10.6f, %10.6f, expected %5.2f %%' % (SOO_poles[jset,n,c],S_poles[jset,n,c],
                                         100*dSdE_poles[jset,n,c]*(EO_poles[jset,n]-EOO_poles[jset,n])/ (S_poles[jset,n,c] - SOO_poles[jset,n,c])))
                    
                    T_mat = LM2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles ) 

                    XSp_mat,XSp_tot,XSp_cap  = T2X_transformsTF(T_mat,CS_diag,gfac,p_mask, n_jsets,n_chans,npairs)
                
                    AxA = T2B_transformsTF(T_mat,AA[:, :,:,:, :,:,:], n_jsets,n_chans,n_angles,batches)
                    AxA = AddCoulombsTF(AxA,  Rutherford, InterferenceAmpl, T_mat, Gfacc, n_angles)
                    
                    AxI = tf.reduce_sum(XSp_mat[n_angle_integrals0:n_totals0,:,:] * ExptAint, [1,2])   # sum over pout,pin
                    AxT = tf.reduce_sum(XSp_tot[n_totals0:n_data,:] * ExptTot, 1)   # sum over pin
                    
                    A_t = tf.concat([AxA, AxI, AxT], 0) 
    
                    chisq = ChiSqTF(A_t, data_val,norm_val,norm_info,effect_norm)
                    print('And ChiSqTF again =',chisq.numpy()/n_data )
                
                optim_results = tfp.optimizer.bfgs_minimize (FitMeasureTF, initial_position=searchpars,
                        max_iterations=Iterations, tolerance=float(Search))
                new_cppt = optim_results.objective_value.numpy()/n_data
                if new_cppt >= last_cppt: break
                last_cppt = new_cppt
                      
            print('\nResults:')
            print('Converged:',optim_results.converged.numpy(), 'Failed:',optim_results.failed.numpy())
            print('Num_iterations:',optim_results.num_iterations.numpy(), 'Num_objective_evaluations:',optim_results.num_objective_evaluations.numpy())
            
            chisqF = optim_results.objective_value
            chisqF_n = chisqF.numpy()
            print('Objective_value:',chisqF_n, 'Objective chisq/pt',chisqF_n/n_data)
            
            searchpars = optim_results.position
            searchpars_n = searchpars.numpy()
            print('position:',searchpars_n)
            
            inverse_hessian = optim_results.inverse_hessian_estimate.numpy()
            print('inverse_hessian: shape=',inverse_hessian.shape ,'\ndiagonal:',[inverse_hessian[i,i] for i in range(n_pars)] )
        
        else:
            inverse_hessian = None
            searchpars_n = searchpars0
        
        print("Wrapup tf: ",tim.toString( ))
        chisqF,A_tF,Grads, T_mat,XSp_mat,XSp_tot,XSp_cap = FitStatusTF(searchpars) 
        chisqF_n = chisqF.numpy()
        A_tF_n = A_tF.numpy()
        grad1 = Grads[0].numpy()
        print(  'chisq from FitStatusTF:',chisqF_n)

        if TransitionMatrix:
            printExcitationFunctions(XSp_tot.numpy(),XSp_cap.numpy(), XSp_mat.numpy(), pname,tname, za,zb, npairs, base+'/'+base,n_data,data_val[:,0],cm2lab,QI,ipair )   

#  END OF TENSORFLOW

    return( searchpars_n, chisqF_n, A_tF_n, grad1, inverse_hessian,  chisq0_n,grad0)
