'''
This file includes utilities to read the kout.txt for all frames.
'''
import sys,os
sys.path.append(os.path.realpath(__file__))
import h5py
import numpy as np
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
import scipy

E_ph=17 #in keV
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m
#k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
#k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
#k_cen=1/wave_len*np.array([-0.03115,-0.02308,0.999248]).reshape(3,1)
k_cen = np.genfromtxt('/home/lichufen/CCB_ind/k_cen.txt')


def get_ind(filename):
    filename=os.path.abspath(filename)
    f=open(filename,'r')
    l=f.readlines()
    ind_bool=np.zeros((len(l),)).astype(bool)
    for m in range(len(l)):
        ind_bool[m]=('#' in l[m])
    ind=np.arange(len(l))[ind_bool]
    ind=np.append(ind,len(l))
    ind=ind-np.arange(len(ind))
    num_frame=ind.shape[0]-1
    f.close()
    return num_frame, ind

def kout_read(filename):
    filename=os.path.abspath(filename)
    num_frame,ind=get_ind(filename)
    kout_dir_dict=dict()
    whole_list=np.genfromtxt(filename,comments='#')
    for frame in range(num_frame):
        vars()['kout_dir_'+str(frame)]=whole_list[ind[frame]:ind[frame+1],:]
        kout_dir_dict['kout_dir_'+str(frame)]=vars()['kout_dir_'+str(frame)]
    return kout_dir_dict

def kout_dir_adj(kout_dir_dict,amp_fact,kosx,kosy):
    l=list(kout_dir_dict)
    num_frame=len(l)
    for lll in range(num_frame):
        ll=l[lll]
        kout_dir=kout_dir_dict[ll]
        kout_dir=kout_dir/kout_dir[:,-1].reshape(-1,1) #normalize by the z-component
        #print(kout_dir)
        kout_dir[:,0:2]=kout_dir[:,0:2]-np.array([kosx,kosy]).reshape(1,2)
        kout_dir[:,0:2]=kout_dir[:,0:2]*amp_fact
        kout_dir_dict[ll]=kout_dir
        #print(kout_dir)
    return kout_dir_dict

def get_kout(kout_dir,E_ph):
    kout_dir_len=np.linalg.norm(kout_dir,axis=1)
    wave_len=12.40/E_ph #in keV and Anstrom
    k0=1/wave_len*1e10 #in m-1
    kout=kout_dir/np.linalg.norm(kout_dir,axis=1).reshape(-1,1)*k0
    return kout

def get_kout_allframe(kout_dir_dict,E_ph):
    l=list(kout_dir_dict)
    num_frame=len(l)
    kout_dict=dict()
    q_dict=dict()
    for lll in range(len(l)):
        ll=l[lll]
        #print('processing'+ll)
        kout_dir=kout_dir_dict[ll]
        kout=get_kout(kout_dir,E_ph)
        #q=kout-np.array([0,0,1e10/(12.40/E_ph)]).reshape(1,3)
        q=kout-k_cen[lll,:].reshape(1,3)
        kout_dict['kout_'+str(lll)]=kout
        q_dict['q_'+str(lll)]=q
    return kout_dict,q_dict
