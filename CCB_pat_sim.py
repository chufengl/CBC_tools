'''
CCB_pat_sim.py simulates the diffraction pattern from a CCB condition.
'''

import sys,os
import numpy as np
import matplotlib
import h5py
import matplotlib.pyplot as plot
import Xtal_calc_util as xu
import CCB_ref
import CCB_pred
import matplotlib.pyplot as plt

E_ph=17
#wave_len= 1e-10*12.40/E_ph
wave_len= 1e-10*12.04/E_ph
k0=1/wave_len
k_in_cen=np.array([0,0,k0]).reshape(3,1)

OR_mat=np.array([[ 4.47536571e+08,-1.33238725e+08,0.00000000e+00],\
[9.38439088e+07,6.35408337e+08,0.00000000e+00],\
[0.00000000e+00,0.00000000e+00,4.00000000e+08]])
def Rot_mat_gen_q(q,alpha):
    q=q.reshape(3,1)
    u=q/np.linalg.norm(q,axis=0)
    ux,uy,uz=u[0],u[1],u[2]
    alpha=np.deg2rad(alpha)
    Rot_mat=np.zeros((3,3))

    Rot_mat[0,0]=np.cos(alpha)+ux**2*(1-np.cos(alpha))
    Rot_mat[0,1]=ux*uy*(1-np.cos(alpha))-uz*np.sin(alpha)
    Rot_mat[0,2]=ux*uz*(1-np.cos(alpha))+uy*np.sin(alpha)
    Rot_mat[1,0]=uy*ux*(1-np.cos(alpha))+uz*np.sin(alpha)
    Rot_mat[1,1]=np.cos(alpha)+uy**2*(1-np.cos(alpha))
    Rot_mat[1,2]=uy*uz*(1-np.cos(alpha))-ux*np.sin(alpha)
    Rot_mat[2,0]=uz*ux*(1-np.cos(alpha))-uy*np.sin(alpha)
    Rot_mat[2,1]=uz*uy*(1-np.cos(alpha))+ux*np.sin(alpha)
    Rot_mat[2,2]=np.cos(alpha)+uz**2*(1-np.cos(alpha))
    return Rot_mat

def get_kins(q,k_in_cen):
    q=q.reshape(3,1)
    k_in_cen=k_in_cen.reshape(3,1)
    n=np.cross(q,k_in_cen,axis=0)
    n_u=n/np.linalg.norm(n,axis=0)
    p_u=np.cross(n_u,q,axis=0)
    p_u=p_u/np.linalg.norm(p_u,axis=0)
    p=np.sqrt(np.linalg.norm(k_in_cen,axis=0)**2-np.linalg.norm(q/2,axis=0)**2)*p_u
    k_in_s=p-q/2
    k_out_s=k_in_s+q
    return k_in_s, k_out_s

def pupil_func(k_in):
    k_in_x=k_in[0]
    k_in_y=k_in[1]
    valid_value=(k_in_x<1.5e8)*(k_in_x>(-1.5e8))*(k_in_y<2.5e8)*(k_in_y>(-2.5e8))
    return valid_value

def source_line_scan(k_in_cen,OR,HKL,rot_ang_step=0.05,rot_ang_range=1.5):
    '''
    source_line_scan
    compute all possible k_in for a centain pupil, from
    an arbitrary first k_in_s determined. This first determined k_in_s
    is usually in the plane of k_in_cen and q vector.
    '''
    q=xu.Get_relp_q(OR,HKL)
    q=q.reshape(3,1)
    K_in_SL=np.array([]).reshape(-1,3)
    K_out_SL=np.array([]).reshape(-1,3)

    k_in_s,k_out_s=get_kins(q,k_in_cen)
    #print('k_in_s',k_in_s)
    #print('k_out_s',k_out_s)

    for ang in np.arange(-rot_ang_range,rot_ang_range,rot_ang_step):
        Rot_mat=Rot_mat_gen_q(q,ang)
        k_in=Rot_mat@k_in_s
        k_out=Rot_mat@k_out_s
        #print('k_in',k_in)
        valid_value=pupil_func(k_in)
        #print(valid_value)
        if valid_value:
            K_in_SL=np.append(K_in_SL,k_in.T,axis=0)
            K_out_SL=np.append(K_out_SL,k_out.T,axis=0)

    return K_in_SL, K_out_SL

def gen_HKL_list(res_cut,OR):
    ## res_cut: resolution cutoff in Angstrom
    ## lp: lattice parameters
    H=np.arange(-50,50,1)
    K=np.arange(-50,50,1)
    L=np.arange(-50,50,1)
    HH,KK,LL=np.meshgrid(H,K,L)
    HH=HH.reshape(-1)
    KK=KK.reshape(-1)
    LL=LL.reshape(-1)
    HKL_list=np.zeros((HH.shape[0],4))
    HKL_list[:,0:3]=np.stack((HH,KK,LL),axis=-1)
    Q=OR@(HKL_list[:,0:3].T)
    Q_len=np.linalg.norm(Q,axis=0)
    ind=(Q_len!=0)
    Q_len=Q_len[ind]
    HKL_list=HKL_list[ind,:]
    HKL_list[:,3]=1e10/Q_len
    ind=np.argsort(HKL_list[:,3])
    ind=ind[-1:0:-1]
    HKL_list=HKL_list[ind,:]
    HKL_list=HKL_list[HKL_list[:,-1]>=res_cut]
    return HKL_list

def pat_sim_q(OR,res_cut):
    ## Pattern_q_table:
    ## col3,4,5: k_in vect
    K_in_table=np.array([]).reshape(-1,3)
    K_out_table=np.array([]).reshape(-1,3)
    HKL_table=np.array([]).reshape(-1,4) # the fourth col is the
    #number of the K vectors in K_in_table and K_out_table from the corresponding
    #HKL.
    HKL_list=gen_HKL_list(res_cut,OR)
    num=HKL_list.shape[0]
    #print(num)
    for l in range(num):
        HKL=HKL_list[l,0:3]
        K_in_SL, K_out_SL=source_line_scan(k_in_cen,OR,HKL,rot_ang_step=0.2,rot_ang_range=1.5)
        #print(K_out_SL.shape)
        if K_in_SL.shape[0]!=0:
            HKL_table=np.append(HKL_table,np.append(HKL,K_in_SL.shape[0]).reshape(1,-1),axis=0)
            K_in_table=np.append(K_in_table,K_in_SL,axis=0)
            K_out_table=np.append(K_out_table,K_out_SL,axis=0)

    return HKL_table, K_in_table, K_out_table

def in_plane_cor(X_L,NA,L0,tilt_ang,K_in_table,K_out_table):
    ## X_L: crystal length
    ## NA: numerical aperturn in k space
    ## L0: nominal/average camera length(usually the value for the center
    ##    of the crystal)
    ## L0 in this case of in_plane tilt, is const
    ## comment: To simulate the control pattern where no CCB tilt-crystal effect is present,
    ## set X_L=0, tilt_angle=0
    L=np.zeros((K_in_table.shape[0],))
    L=L0
    X=X_L/NA*np.sin(np.deg2rad(tilt_ang))*K_in_table[:,0]+K_out_table[:,0]/K_out_table[:,2]*L
    Y=X_L/NA*np.cos(np.deg2rad(tilt_ang))*K_in_table[:,1]+K_out_table[:,1]/K_out_table[:,2]*L
    XY=np.stack((X,Y),axis=-1)

    return XY

def off_plane_cor(X_L,NA,L0,tilt_ang,K_in_table,K_out_table):
    ## the crystal tilt is assumed to be in the y-z plane.
    ## x=0 const,

    L=L0+X_L/NA*np.sin(np.deg2rad(tilt_ang))*K_in_table[:,1]
    print(L)
    X=0+K_out_table[:,0]/K_out_table[:,2]*L
    Y=X_L/NA*np.cos(np.deg2rad(tilt_ang))*K_in_table[:,1]+K_out_table[:,1]/K_out_table[:,2]*L
    XY=np.stack((X,Y),axis=-1)

    return XY

def XY2P(XY,pix_size,px0,py0):
    PX=XY[:,0]/pix_size+px0
    PY=XY[:,1]/pix_size+py0
    PXY=np.stack((PX,PY),axis=-1)
    return PXY
