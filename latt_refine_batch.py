'''
latt_refine_batch.py does the lattice refinement in a batch,
after the DE refinement.
'''
import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
import matplotlib
import h5py
import matplotlib.pyplot as plot
import Xtal_calc_util as xu
import CCB_ref
import CCB_pred
import CCB_read
import CCB_pat_sim
import matplotlib.pyplot as plt
import scipy.optimize
import batch_refine
import gen_match_figs as gm

a=15.4029218
b=21.86892773
c=25
Alpha=90
Beta=90
Gamma=90
OR_mat=np.array([[ 4.47536571e+08,-1.33238725e+08,0.00000000e+00],\
[9.38439088e+07,6.35408337e+08,0.00000000e+00],\
[0.00000000e+00,0.00000000e+00,4.00000000e+08]])
OR_mat=OR_mat/1.03

E_ph=17 #in keV
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m
k_cen=np.array([0,0,1/wave_len]).reshape(3,1)

def latt_frame_refine(ind1,res_file):
    res_arry=gm.read_res(res_file)
    frame=int(res_arry[ind1,0])
    OR_angs=tuple(res_arry[ind1,1:4])
    cam_len=res_arry[ind1,4]
    k_out_osx=res_arry[ind1,5]
    k_out_osy=res_arry[ind1,6]
    theta,phi,alpha=OR_angs
    OR=CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(-frame)@OR_mat
    OR=OR*1e-8
    k_out_osx=k_out_osx*1e2
    K_out_osy=k_out_osy*1e2
    x0=tuple(np.concatenate((OR.reshape(-1,),np.array([cam_len,k_out_osx,k_out_osy])),axis=-1))
    x0_GA=tuple(res_arry[ind1,1:7])
    args=(frame,x0_GA)
    #print('Refining OR for frame %d'%(frame))
    res = scipy.optimize.minimize(CCB_ref._TG_func6, x0, args=args, method='CG', options={'disp': True})
    return res


def latt_batch_refine(res_file):
    f=open('Latt_refine.txt','a',1)
    f.write('The GA_refine res_file is:\n%s\n'%(os.path.abspath(res_file)))
    f.write('====================================\n')
    res_arry=gm.read_res(res_file)
    print('res_file: %s'%(os.path.abspath(res_file)))
    print('%d frames loaded in the res_file'%(res_arry.shape[0]))
    for ind1 in range(res_arry.shape[0]):
        frame=int(res_arry[ind1,0])
        print('Lattice Refining frame %03d'%(frame))
        res=latt_frame_refine(ind1,res_file)
        f.write('frame %03d \n'%(frame))
        f.write('TG before Lattice refinement: %7.3e \n'%(res_arry[ind1,-1]))
        f.write('TG after Lattice refinement: %7.3e \n'%(res.fun))
        f.write('res: \n')
        #f.write('%7.3e %7.3e %7.3e %7.3e %7.3e %7.3e\n'%(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5]))
        f.write('%7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e\n'%(res.x[0]*1e8,res.x[1]*1e8,res.x[2]*1e8,res.x[3]*1e8,res.x[4]*1e8,res.x[5]*1e8,res.x[6]*1e8,res.x[7]*1e8,res.x[8]*1e8,res.x[9],res.x[10]*1e-2,res.x[11]*1e-2))
        f.write('------------------------------------\n')
        print('Done!')
    f.close()
    return

if __name__=='__main__':
    res_file=os.path.abspath(sys.argv[1])
    latt_batch_refine(res_file)
