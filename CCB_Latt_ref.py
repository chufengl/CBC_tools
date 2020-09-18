import sys,os
sys.path.append('/home/lichufen/CCB_ind/scripts/')
import numpy as np 
import batch_refine
OR_mat=np.array([[ 4.47536571e+08,-1.33238725e+08,0.00000000e+00],\
[9.38439088e+07,6.35408337e+08,0.00000000e+00],\
[0.00000000e+00,0.00000000e+00,4.00000000e+08]]) 
OR_mat=OR_mat/1.03                                                      

x0 = OR_mat.T.reshape(-1,)                                              
x_l = x0-4e7                                                            
x_h = x0+4e7                                                            
bounds = tuple([(x_l[m],x_h[m]) for m in range(9)])                     
res = batch_refine.Latt_refine(np.arange(0,100),x0,bounds,'../K_ref_test2/Best_GA_res.txt')      
