'''
CCB_K_int_viewer.py is to diplay the pupil funciton
'''
import sys,os
import numpy as np
import matplotlib.pyplot as plt



def read_kout_int(kout_file,bins):
	k_exp=np.genfromtxt(kout_file)
	bins_arry_x=np.linspace(-10e8,10e8,bins+1)
	bins_arry_y=np.linspace(-10e8,10e8,bins+1)
	bins_ind_x=np.digitize(k_exp[:,-3],bins_arry_x)
	bins_ind_y=np.digitize(k_exp[:,-2],bins_arry_y)
	Int_arry=np.zeros((bins_arry_x.shape[0],bins_arry_y.shape[0]))
	for m in range(k_exp[:,-2].shape[0]):
		Int_arry[bins_ind_x[m]-1,bins_ind_y[m]-1]+=k_exp[m,3]
	KX,KY=np.meshgrid(bins_arry_x,bins_arry_y)
	plt.figure();plt.pcolor(KX,KY,Int_arry,cmap='jet');plt.clim(0,5e4);
	plt.colorbar();
	#plt.show()
	plt.savefig('Int_'+os.path.basename(kout_file).split('.')[0]+'.png')
	return None
if __name__=='__main__':
	kout_file=os.path.abspath(sys.argv[1])
	bins=int(sys.argv[2])
	read_kout_int(kout_file,bins)
	print(kout_file+'Done!')
