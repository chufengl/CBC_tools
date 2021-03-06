'''
CCB_pat_sim.py simulates the diffraction pattern from a CCB condition.
'''

import sys,os
import numpy as np
import h5py
import Xtal_calc_util as xu
import CCB_ref
import CCB_pred
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plot

E_ph=17
wave_len= 1e-10*12.40/E_ph
k0=1/wave_len
#k_in_cen=np.array([0,0,k0]).reshape(3,1)
#k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
#k_cen=k0*np.array([-0.03115,-0.02308,0.999248]).reshape(3,1)
k_cen = np.genfromtxt('/home/lichufen/CCB_ind/k_cen.txt')
k_in_cen=k_cen

OR_mat=np.array([[ 4.47536571e+08,-1.33238725e+08,0.00000000e+00],\
[9.38439088e+07,6.35408337e+08,0.00000000e+00],\
[0.00000000e+00,0.00000000e+00,4.00000000e+08]])
OR_mat=OR_mat/1.03
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
    k_in_x=k_in_x - (-0e8)
    k_in_y=k_in_y - (-0e8)
    #valid_value=(k_in_x<8e8)*(k_in_x>-8e8)*(k_in_y<8e8)*(k_in_y>-8e8)
    #valid_value=(k_in_x<0e8)*(k_in_x>-5e8)*(k_in_y<0e8)*(k_in_y>-5e8)
    valid_value=(k_in_x<-2e8)*(k_in_x>-6e8)*(k_in_y<-0e8)*(k_in_y>-7e8)
    return valid_value

def source_line_scan(k_in_cen,OR,HKL,rot_ang_step=0.05,rot_ang_range=3.0):
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
    H=np.arange(-25,25,1)
    K=np.arange(-25,25,1)
    L=np.arange(-25,25,1)
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

def pat_sim_q(k_in_cen,OR,res_cut):
    ## Pattern_q_table:
    ## col3,4,5: k_in vect
    K_in_table=np.array([]).reshape(-1,3)
    K_out_table=np.array([]).reshape(-1,3)
    HKL_table=np.array([]).reshape(-1,4) # the fourth col is the
    #number of the K vectors in K_in_table and K_out_table from the corresponding
    #HKL.
    HKL_list=gen_HKL_list(res_cut,OR)
    num=HKL_list.shape[0]
    for l in range(num):
        HKL=HKL_list[l,0:3]
        K_in_SL, K_out_SL=source_line_scan(k_in_cen,OR,HKL,rot_ang_step=0.03,rot_ang_range=1.5)
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

############################################
## add the functions for Xtal model to address the diffration efficency

def xtal_model_init(xyz_range,voxel_size=5e-6):
    '''
    generate a xtal_model in a python dictionary form. This init
    :param xyz_range: the range of the crystal in x,y,z dimensions, in m.
    	              6-element list,tuple: x_min, x_max, y_min, y_max, z_min, z_max,
    :type xyz_range:  6-element list,tuple
    :param voxel_size: size of the voxel in crystal model,in m. default value 5e-6(5um)
    :type voxel_size:  float
    :return:          xtal_model_dict: x0,y0,z0, as the coordinates of crystal voxels. D, as the
    	              diffraction power.
    :rtype:           python dictionary.
    '''
    x0_arry = np.linspace(xyz_range[0],xyz_range[1],np.rint((xyz_range[1]-xyz_range[0])/voxel_size).astype(np.int)+1)
    y0_arry = np.linspace(xyz_range[2],xyz_range[3],np.rint((xyz_range[3]-xyz_range[2])/voxel_size).astype(np.int)+1)
    z0_arry = np.linspace(xyz_range[4],xyz_range[5],np.rint((xyz_range[5]-xyz_range[4])/voxel_size).astype(np.int)+1)
    #print(x0_arry.shape)
    x0 = []
    y0 = []
    z0 = []
    D = []
    xtal_cen = [(xyz_range[1]+xyz_range[0])/2,(xyz_range[3]+xyz_range[2])/2,(xyz_range[5]+xyz_range[4])/2]
    for x_p in x0_arry:
        for y_p in y0_arry:
            for z_p in z0_arry:
                x0.append(x_p)
                y0.append(y_p)
                z0.append(z_p)
                d = np.exp(-(x_p-xtal_cen[0])**2/(xyz_range[1]-xyz_range[0])**2-(y_p-xtal_cen[1])**2/(xyz_range[3]-xyz_range[2])**2-(z_p-xtal_cen[2])**2/(xyz_range[5]-xyz_range[4])**2)
                D.append(d)  ## binary model as the first try.
    x0 = np.array(x0)
    y0 = np.array(y0)
    z0 = np.array(z0)
    D = np.array(D)
    xtal_model0_dict = {'x0':x0, 'y0':y0, 'z0':z0, 'D':D}
    return xtal_model0_dict

def k_in_render(xtal_model0_dict,rot_mat,pivot_coor,focus_coor=[0,0,-0.129]):
    '''
    calculate the k_in for a given orienation/ rotation matrix, pivot point of crystal rotation
    and focal point position of the MLL set.
    '''
    x0 = xtal_model0_dict['x0']
    y0 = xtal_model0_dict['y0']
    z0 = xtal_model0_dict['z0']
    D = xtal_model0_dict['D']
	
    xyz0 = np.vstack((x0.reshape(1,-1),y0.reshape(1,-1),z0.reshape(1,-1)))
    xyz = rot_mat@xyz0 + (np.identity(3) - rot_mat)@(np.array(pivot_coor).reshape(3,1))
    x = xyz[0,:]
    y = xyz[1,:]
    z = xyz[2,:]
	
    k_in = xyz - np.array(focus_coor).reshape(3,1)
    k_in = k_in/np.linalg.norm(k_in,axis=0)*k0# the k_in shape (3,N)
    xtal_model_dict = {'x':x,'y':y,'z':z,'D':D,'k_in':k_in}
    return xtal_model_dict

def get_D(xtal_model_dict,k_in,delta_k_in=1e7):
    '''
    computes the D_value for given *xtal_model_dict* and k_in.
    D_value:
        the sum of the voxel values of all voxels which have the given k_in, within
        the delta_k_in range bin.
    '''
    D_arry = xtal_model_dict['D']
    k_in_arry = xtal_model_dict['k_in']
    ind = ((k_in_arry.T>=(k_in-delta_k_in))*(k_in_arry.T<(k_in+delta_k_in))).all(axis=1).nonzero()[0]
    D_value = D_arry[ind].sum()
    return D_value

def get_P(ref_image,k_in):
    '''
    computes the P_value for given reference image(which consists of pupil shade in the middle)
    and k_in.
    P_value:
        the pupil value that corresponds to the given k_in.
    
    This function depends on **Scattering geometry** which might vary for different experiments and data sets.
    For now, the geometrical parameters are hard coded for CBC B12 scan135 data set.
    '''
    x_ind, y_ind = (k_in/k_in[2]*0.129)[0:2]            # distance between focus and detector 0.129m
    x_ind = np.rint((x_ind/75e-6)+1594).astype(np.int)  # the x coordinate of the forward beam on detector.
    y_ind = np.rint((y_ind/75e-6)+1764).astype(np.int)  # the y coordiante of the forward beam on detector.
    pu = ref_image[y_ind-3:y_ind+3,x_ind-3:x_ind+3].mean()/1e6                     # the normalised pixel value of pupil shade for k_in. no mean is taken here.
    if (pu>1):
        P_value = pu
    else:
        P_value = 0
    return P_value

def get_Int_ref(hkl_file):
    Int_ref_arry = np.genfromtxt(hkl_file,skip_header=3,skip_footer=2,usecols=(0,1,2,3))
    Int_ref_arry = np.hstack((Int_ref_arry[:,1:2],Int_ref_arry[:,0:1],Int_ref_arry[:,2:]))
    return Int_ref_arry

def compt_Int_sim(Int_ref_arry,HKL,P_value,D_value):
    '''
    computes simulated reflection intensity for given reflection list, HKL, P_value, and D_value
    this function is designed to take imtermediate variables to be more of general use.
    '''
	
    HKL = np.array(HKL).reshape(-1,)
    ind = (Int_ref_arry[:,:3]==np.abs(HKL)).all(axis=1).nonzero()[0]
    if ind.shape[0]==1:
        Int_ref = Int_ref_arry[ind,3]
    else:
        Int_ref = np.nan
        print('HKL not found in reflection list',HKL)
    #print('D_value:',D_value,'P_value',P_value,'Int_ref',Int_ref)
    Int_sim = D_value*Int_ref*P_value
    return Int_sim
