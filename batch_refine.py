'''
batch_refine.py does the refinement in a batch.
'''
import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
import matplotlib
import h5py
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import Xtal_calc_util as xu
import CCB_ref
import CCB_pred
import CCB_read
import CCB_pat_sim
import matplotlib.pyplot as plt
import scipy.optimize


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

def point_match(frame,OR,amp_fact,kosx,kosy,E_ph):
    #frame=0
    #amp_fact=1
    #kosx,kosy=0,0
    #OR=CCB_ref.rot_mat_zaxis(0)@CCB_ref.rot_mat_xaxis(0)@CCB_ref.rot_mat_yaxis(-frame+0)@OR_mat


    #amp_fact=res.x[-3]
    #kosx,kosy=res.x[-2],res.x[-1]
    #OR=CCB_ref.Rot_mat_gen(res.x[0],res.x[1],res.x[2])@CCB_ref.rot_mat_yaxis(-frame)@OR_mat

    #amp_fact=1.008
    #kosx,kosy=3.614829e-2,5.833e-3
    #OR=CCB_ref.Rot_mat_gen(6.1267e1,-9.697e1,2.424e0)@CCB_ref.rot_mat_yaxis(-frame)@OR_mat

    #OR=Rot_mat@OR
    #OR=x_arry[-1,0:9].reshape(3,3)
    #OR=OR_V
    #OR=OR_refd.reshape(3,3)
    #print(CCB_ref.rot_mat_yaxis(-frame+0)@OR_mat)
    #print(OR)
    E_ph=17
    wave_len=1e-10*12.40/E_ph
    frac_offset=np.array([0,0,0])

    kout_dir_dict=CCB_read.kout_read('/home/lichufen/CCB_ind/k_out.txt')
    kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact,kosx,kosy)
    kout_dict,q_dict=CCB_read.get_kout_allframe(kout_dir_dict,E_ph)
    Q_arry=q_dict['q_'+str(frame)]
    K_out=kout_dict['kout_'+str(frame)]
    #HKL_frac, HKL_int, Q_int, Q_resid = CCB_ref.get_HKL(OR,Q_arry,np.array([0,0,0]))
    frac_offset=np.array([0,0,0])
    HKL_frac, HKL_int, Q_int, Q_resid = CCB_ref.get_HKL8(OR,Q_arry,frac_offset)
    Delta_k, Dist, Dist_1=CCB_ref.exctn_error8_nr(OR,Q_arry,Q_int,frac_offset,E_ph)

    ind=np.argsort(Dist,axis=1)

    ind=np.array([ind[m,0] for m in range(ind.shape[0])])
    Dist=np.array([Dist[m,ind[m]] for m in range(Dist.shape[0])])
    HKL_int=np.array([HKL_int[m,:,ind[m]] for m in range(HKL_int.shape[0])])
    Delta_k=np.array([Delta_k[m,:,ind[m]] for m in range(Delta_k.shape[0])])



    K_in_pred,K_out_pred=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_int)
    Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3)
    Delta_k_out_new=K_out_pred-K_out

    #K_in_pred,K_out_pred=CCB_pred.kout_pred8(OR,[0,0,1/wave_len],HKL_int)
    #Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3,1)
    #Delta_k_out_new=K_out_pred-K_out.reshape(-1,3,1)
    #Dist2=np.linalg.norm(Delta_k_out_new,axis=1)
    #ind2=np.argsort(Dist2,axis=1)
    #ind2=np.array([ind2[m,0] for m in range(ind2.shape[0])])
    #Dist2=np.array([Dist2[m,ind2[m]] for m in range(Dist2.shape[0])])
    #HKL_int=np.array([HKL_int[m,:,ind2[m]] for m in range(HKL_int.shape[0])])
    #Delta_k=np.array([Delta_k[m,:,ind2[m]] for m in range(Delta_k.shape[0])])
    #K_in_pred=np.array([K_in_pred[m,:,ind2[m]] for m in range(K_in_pred.shape[0])])
    #K_out_pred=np.array([K_out_pred[m,:,ind2[m]] for m in range(K_out_pred.shape[0])])
    #Delta_k_in_new=np.array([Delta_k_in_new[m,:,ind2[m]] for m in range(Delta_k_in_new.shape[0])])
    #Delta_k_out_new=np.array([Delta_k_out_new[m,:,ind2[m]] for m in range(Delta_k_out_new.shape[0])])


    ind_filter_1=np.linalg.norm(Delta_k_out_new,axis=1)<10e8
    ind_filter_2=np.linalg.norm(Delta_k_in_new,axis=1)<10e8
    ind_filter=ind_filter_1*ind_filter_2
    Delta_k_in_new=Delta_k_in_new[ind_filter,:]
    Delta_k_out_new=Delta_k_out_new[ind_filter,:]
    K_out=K_out[ind_filter,:]
    K_out_pred=K_out_pred[ind_filter,:]
    K_in_pred=K_in_pred[ind_filter,:]

    #plt.figure()
    #plt.scatter(Delta_k_in_new[:,0],Delta_k_in_new[:,1],s=1,marker='x',color='b')
    #plt.axis('equal')
    #plt.xticks(np.linspace(-5e8,5e8,5));
    #plt.yticks(np.linspace(-5e8,5e8,5));
    #plt.xlim(-5e8,5e8)
    #plt.ylim(-5e8,5e8)

    #plt.figure()
    #plt.scatter(Delta_k_out_new[:,0],Delta_k_out_new[:,1],s=1,marker='x',color='b')
    #plt.axis('equal')
    #plt.xticks(np.linspace(-5e8,5e8,5));
    #plt.yticks(np.linspace(-5e8,5e8,5));
    #plt.xlim(-5e8,5e8)
    #plt.ylim(-5e8,5e8)


    #plt.figure(figsize=(10,10))
    #plt.scatter(K_out[:,0],K_out[:,1],s=1,marker='x',color='b')
    #plt.scatter(K_out_pred[:,0],K_out_pred[:,1],s=1,marker='x',color='r')
    #plt.axis('equal')
    #plt.savefig('point_match_frame%03d'%(frame)+'.png')
    #plt.close('all')

    return K_out, K_in_pred, K_out_pred

def GA_refine(frame,bounds):
    args=(frame,)
    #bounds=((0,90),(-180,180),(-6,6),(0.95,1.05),(-5e-2,5e-2),(-5e-2,5e-2),(-0.1,0.1),(-0.1,0.1),(-0.1,0.1),(-3,3),(-3,3),(-3,3))
    res = scipy.optimize.differential_evolution(CCB_ref._TG_func3,bounds,args=args,strategy='best1bin',disp=True,polish=True)
    #res = scipy.optimize.differential_evolution(CCB_ref._TG_func5,bounds,args=args,strategy='best1bin',disp=True,polish=True)
    print('intial','TG: %7.3e'%CCB_ref._TG_func3(np.array([0,0,0,1,0,0]),frame))
    print('final',res.x,'TG: %7.3e'%CCB_ref._TG_func3(res.x,frame))
    #print('intial','TG: %7.3e'%CCB_ref._TG_func5(np.array([0,0,0,1,0,0,a,b,c,Alpha,Beta,Gamma]),frame))
    #print('final',res.x,'TG: %7.3e'%CCB_ref._TG_func5(res.x,frame))
    return res

def frame_refine(frame,res_cut=1,E_ph=17):
    wave_len= 1e-10*12.40/E_ph
    k0=1/wave_len
    k_in_cen=np.array([0,0,k0]).reshape(3,1)
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


    amp_fact=1
    kosx,kosy=0,0
    OR=CCB_ref.rot_mat_yaxis(-frame+0)@OR_mat
    K_out, K_in_pred, K_out_pred=point_match(frame,OR,amp_fact,kosx,kosy,E_ph)
    HKL_table, K_in_table, K_out_table=CCB_pat_sim.pat_sim_q(OR,res_cut)
    K_in_pred_s,K_out_pred_s=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_table[:,0:3])
    plt.figure(figsize=(10,10))
    plt.scatter(K_out_table[:,0],K_out_table[:,1],s=1,marker='x',c='g')
    plt.scatter(K_out[:,0],K_out[:,1],s=20,marker='x',color='b')
    plt.scatter(K_out_pred[:,0],K_out_pred[:,1],s=20,marker='x',color='r')
    plt.scatter(K_out_pred_s[:,0],K_out_pred_s[:,1],s=40,marker='o',edgecolor='black',facecolor='None')
    plt.axis('equal')
    plt.savefig('line_match_before_frame%03d.png'%(frame))

    bounds=((0,90),(-180,180),(-5,5),(0.95,1.00),(-0.0e-2,0.0e-2),(-0e-2,0e-2))
    res=GA_refine(frame,bounds)
    #f.write('frame %03d \n'%(frame))
    #f.write('intial','TG: %7.3e'%CCB_ref._TG_func3(np.array([0,0,0,1,0,0]),frame))
    #f.write('final',res.x,'TG: %7.3e'%CCB_ref._TG_func3(res.x,frame))
    #f.write('------------------------------------\n')
    amp_fact=res.x[3]
    kosx,kosy=res.x[4],res.x[5]
    #lp=np.array([1e-10*res.x[6],1e-10*res.x[7],1e-10*res.x[8],res.x[9],res.x[10],res.x[11]])
    #_,OR_mat=xu.A_gen(lp)
    #OR_start=CCB_ref.rot_mat_xaxis(0)@CCB_ref.rot_mat_yaxis(-frame)@CCB_ref.rot_mat_zaxis(11.84)@OR_mat
    #OR=CCB_ref.Rot_mat_gen(res.x[0],res.x[1],res.x[2])@OR_start
    OR=CCB_ref.Rot_mat_gen(res.x[0],res.x[1],res.x[2])@CCB_ref.rot_mat_yaxis(-frame)@OR_mat
    K_out, K_in_pred, K_out_pred=point_match(frame,OR,amp_fact,kosx,kosy,E_ph)
    HKL_table, K_in_table, K_out_table=CCB_pat_sim.pat_sim_q(OR,res_cut)
    K_in_pred_s,K_out_pred_s=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_table[:,0:3])



    plt.figure(figsize=(10,10))
    plt.scatter(K_out_table[:,0],K_out_table[:,1],s=1,marker='x',c='g')
    plt.scatter(K_out[:,0],K_out[:,1],s=20,marker='x',color='b')
    plt.scatter(K_out_pred[:,0],K_out_pred[:,1],s=20,marker='x',color='r')
    plt.scatter(K_out_pred_s[:,0],K_out_pred_s[:,1],s=40,marker='o',edgecolor='black',facecolor='None')
    plt.axis('equal')
    plt.savefig('line_match_after_frame%03d.png'%(frame))
    plt.close('all')
    #plt.figure(figsize=(10,10))
    #plt.scatter(K_in_table[:,0],K_in_table[:,1],s=4,marker='o',c='g')
    #plt.scatter(K_in_pred_s[:,0],K_in_pred_s[:,1],s=4,marker='o',c='black')
    #plt.scatter(K_in_pred[:,0],K_in_pred[:,1],s=4,marker='o',c='r')

    #plt.figure(figsize=(10,10))
    #plt.scatter(Delta_k_in_new[:,0],Delta_k_in_new[:,1],s=10,marker='o',c=np.linalg.norm(Delta_k_out_new,axis=1),cmap='jet')
    #plt.colorbar()

    return res

def batch_refine(start_frame,end_frame):
    f=open('GA_refine.txt','a',1)

    for frame in range(start_frame,end_frame+1):
        print('Refining frame %03d'%(frame))
        res=frame_refine(frame,res_cut=1,E_ph=17)
        f.write('frame %03d \n'%(frame))
        f.write('intial TG: %7.3e \n'%CCB_ref._TG_func3(np.array([0,0,0,1,0,0]),frame))
        f.write('final TG: %7.3e \n'%CCB_ref._TG_func3(res.x,frame))
        f.write('res: \n')
        f.write('%7.3e %7.3e %7.3e %7.3e %7.3e %7.3e\n'%(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5]))
        #f.write('%7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e\n'%(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],res.x[6],res.x[7],res.x[8],res.x[9],res.x[10],res.x[11]))
        f.write('------------------------------------\n')
    f.close()
    return

if __name__=='__main__':
    start_frame=int(sys.argv[1])
    end_frame=int(sys.argv[2])
    batch_refine(start_frame,end_frame)
