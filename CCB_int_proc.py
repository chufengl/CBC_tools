'''
CCB_int_proc.py 

processes the intensities of steaks and pixels for frames and data set.

'''

import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
from skimage import measure, morphology, feature
import scipy
import glob
import h5py
import re
import CCB_ref
import CCB_pred
import CCB_pat_sim
import CCB_read
#import gen_match_figs as gm
import CCB_streak_det
import matplotlib
#matplotlib.use('TkAgg') # To be adjusted for the batch job mode.
import matplotlib.pyplot as plt
import h5py
import scipy.ndimage as ndi
import pickle as pk


E_ph = 17
wave_len = 1e-10*12.4/E_ph
k0 = 1/wave_len


class Dataset:
	
	def __init__(self,K_map_file):
		self.K_map_file = os.path.abspath(K_map_file)
		self.arry = np.genfromtxt(K_map_file)
		self.get_frames()
		self.clean()
		self.get_hkl_arry()
		self.get_HKL_arry()
		self.adj_frame()
		
		
	def get_frames(self):
		frame_arry, frame_ind = np.unique(self.arry[:,0], return_inverse=True)
		self.frame_total_no = frame_arry.shape[0]
		self.frame_arry = frame_arry.astype(np.int)
		self.frame_obj_list=[]
		for i in range(self.frame_total_no):
			frame = Frame(self, self.frame_arry[i])               # create the Frame obj.
			self.frame_obj_list.append(frame)
		
	
	def get_HKL_arry(self):
		HKL_arry=np.array([],dtype=np.int).reshape(-1,3)
		for f in self.frame_obj_list:
			hkl_arry = np.abs(f.hkl_arry)
			HKL_arry = np.vstack((HKL_arry, hkl_arry))
		HKL_arry, ind = np.unique(HKL_arry,axis=0,return_inverse=True)
		Redundancy=np.array([np.count_nonzero(ind==m) for m in range(HKL_arry.shape[0])])
		self.HKL_arry = HKL_arry
		self.Redundancy = Redundancy
	
	def get_hkl_arry(self):
		hkl_arry = np.array([],dtype=np.int).reshape(-1,3)
		for f in self.frame_obj_list:
			hkl_arry = np.vstack((hkl_arry, f.hkl_arry))
		hkl_arry, ind = np.unique(hkl_arry, axis=0, return_inverse=True)
		redundancy = np.array([np.count_nonzero(ind==m) for m in range(hkl_arry.shape[0])])
		self.hkl_arry = hkl_arry
		self.redundancy = redundancy
		
	def search_hkl(self, hkl):
		frame_ar = np.array([],dtype=np.int)
		inte_ave_ar = np.array([])
		inte_sum_ar = np.array([])
		inte_ave_adj_ar = np.array([])
		inte_sum_adj_ar = np.array([])
		pixel_total_no_ar = np.array([])
		r_obj_list=[]
		for f in self.frame_obj_list:
			frame = f.frame_no
			r = f.find_hkl(hkl)
			if r is not None:
				frame_ar = np.append(frame_ar, frame)
				inte_ave_ar = np.append(inte_ave_ar, r.ref_inte_ave)
				inte_sum_ar = np.append(inte_sum_ar, r.ref_inte_sum)
				inte_ave_adj_ar = np.append(inte_ave_adj_ar, r.ref_inte_ave_adj)
				inte_sum_adj_ar = np.append(inte_sum_adj_ar, r.ref_inte_sum_adj)
				pixel_total_no_ar = np.append(pixel_total_no_ar, r.pixel_total_no)
				r_obj_list.append(r)
		return {'frame_ar':frame_ar, 'inte_ave_ar':inte_ave_ar, 'inte_sum_ar':inte_sum_ar, 'pixel_total_no_ar':pixel_total_no_ar,\
					'r_obj_list':r_obj_list, 'inte_ave_adj_ar':inte_ave_adj_ar, 'inte_sum_adj_ar':inte_sum_adj_ar}

	def search_HKL(self, HKL):
		frame_ar = np.array([],dtype=np.int)
		inte_ave_ar = np.array([])
		inte_sum_ar = np.array([])
		inte_ave_adj_ar = np.array([])
		inte_sum_adj_ar = np.array([])
		pixel_total_no_ar = np.array([])
		r_obj_list=[]
		for f in self.frame_obj_list:
			frame = f.frame_no
			r_l = f.find_HKL(HKL)
			#print(len(r_l))
			if len(r_l)!=0:
				for r in r_l:
					frame_ar = np.append(frame_ar, frame)
					inte_ave_ar = np.append(inte_ave_ar, r.ref_inte_ave)
					inte_sum_ar = np.append(inte_sum_ar, r.ref_inte_sum)
					inte_ave_adj_ar = np.append(inte_ave_adj_ar, r.ref_inte_ave_adj)
					inte_sum_adj_ar = np.append(inte_sum_adj_ar, r.ref_inte_sum_adj)
					pixel_total_no_ar = np.append(pixel_total_no_ar, r.pixel_total_no)
					r_obj_list.append(r)
		return {'frame_ar':frame_ar, 'inte_ave_ar':inte_ave_ar, 'inte_sum_ar':inte_sum_ar, 'pixel_total_no_ar':pixel_total_no_ar,\
				'r_obj_list':r_obj_list, 'inte_ave_adj_ar':inte_ave_adj_ar, 'inte_sum_adj_ar':inte_sum_adj_ar}
	
	def adj_frame(self):
		for f in self.frame_obj_list:
			f.adj_k_in()
			for r in f.reflection_obj_list:
				r.get_inte_adj()
		

	def merge_all_hkl(self):
		out_put_arry=np.zeros((self.hkl_arry.shape[0],16))
		for m in range(self.hkl_arry.shape[0]):
			hkl = self.hkl_arry[m,:]
			dd = self.search_hkl(hkl)
			out_put_arry[m,0:3] = hkl
			out_put_arry[m,3] = dd['inte_ave_ar'].shape[0]
			out_put_arry[m,4] = dd['inte_ave_ar'].mean()
			#out_put_arry[m,4] = dd['inte_ave_ar'].mean()
			out_put_arry[m,5] = dd['inte_ave_ar'].std()
			out_put_arry[m,6] = out_put_arry[m,4]/out_put_arry[m,5]
			out_put_arry[m,7] = dd['inte_sum_ar'].mean()
			out_put_arry[m,8] = dd['inte_sum_ar'].std()
			out_put_arry[m,9] = out_put_arry[m,7]/out_put_arry[m,8]
			out_put_arry[m,10] = dd['inte_ave_adj_ar'].mean()
			out_put_arry[m,11] = dd['inte_ave_adj_ar'].std()
			out_put_arry[m,12] = out_put_arry[m,10]/out_put_arry[m,11]
			out_put_arry[m,13] = dd['inte_sum_adj_ar'].mean()
			out_put_arry[m,14] = dd['inte_sum_adj_ar'].std()
			out_put_arry[m,15] = out_put_arry[m,13]/out_put_arry[m,14]
			print('%d out of %d hkl done!'%(m+1,self.hkl_arry.shape[0]))
		np.savetxt('all_hkl.txt',out_put_arry,fmt=['%4d','%4d','%4d','%03d','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f'])
		return
	
	def merge_all_HKL(self):
		out_put_arry=np.zeros((self.hkl_arry.shape[0],16))
		for m in range(self.HKL_arry.shape[0]):
			HKL = self.HKL_arry[m,:]
			dd = self.search_HKL(HKL)
			out_put_arry[m,0:3] = HKL
			out_put_arry[m,3] = dd['inte_ave_ar'].shape[0]
			out_put_arry[m,4] = dd['inte_ave_ar'].mean()
			out_put_arry[m,5] = dd['inte_ave_ar'].std()
			out_put_arry[m,6] = out_put_arry[m,4]/out_put_arry[m,5]
			out_put_arry[m,7] = dd['inte_sum_ar'].mean()
			out_put_arry[m,8] = dd['inte_sum_ar'].std()
			out_put_arry[m,9] = out_put_arry[m,7]/out_put_arry[m,8]
			out_put_arry[m,10] = dd['inte_ave_adj_ar'].mean()
			out_put_arry[m,11] = dd['inte_ave_adj_ar'].std()
			out_put_arry[m,12] = out_put_arry[m,10]/out_put_arry[m,11]
			out_put_arry[m,13] = dd['inte_sum_adj_ar'].mean()
			out_put_arry[m,14] = dd['inte_sum_adj_ar'].std()
			out_put_arry[m,15] = out_put_arry[m,13]/out_put_arry[m,14]
			print('%d out of %d HKL done!'%(m+1,self.HKL_arry.shape[0]))
		np.savetxt('all_HKL.txt',out_put_arry,fmt=['%4d','%4d','%4d','%03d','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f'])
		return

	def merge_all_HKL_crystfel(self):
		out_put_arry=np.zeros((self.HKL_arry.shape[0],16))
		for m in range(self.HKL_arry.shape[0]):
			HKL = self.HKL_arry[m,:]
			dd = self.search_HKL(HKL)
			out_put_arry[m,0:3] = HKL
			out_put_arry[m,3] = dd['inte_ave_ar'].shape[0]
			out_put_arry[m,4] = np.nanmean(dd['inte_ave_ar'])
			out_put_arry[m,5] = np.nanstd(dd['inte_ave_ar'])
			out_put_arry[m,6] = out_put_arry[m,4]/out_put_arry[m,5]
			out_put_arry[m,7] = np.nanmean(dd['inte_sum_ar'])
			out_put_arry[m,8] = np.nanstd(dd['inte_sum_ar'])
			out_put_arry[m,9] = out_put_arry[m,7]/out_put_arry[m,8]
			out_put_arry[m,10] = np.nanmean(dd['inte_ave_adj_ar'])
			out_put_arry[m,11] = np.nanstd(dd['inte_ave_adj_ar'])
			out_put_arry[m,12] = out_put_arry[m,10]/out_put_arry[m,11]
			out_put_arry[m,13] = np.nanmean(dd['inte_sum_adj_ar'])
			out_put_arry[m,14] = np.nanstd(dd['inte_sum_adj_ar'])
			out_put_arry[m,15] = out_put_arry[m,13]/out_put_arry[m,14]
			print('%d out of %d HKL done!'%(m+1,self.HKL_arry.shape[0]))
		######################
		## output the array in CrystFEL reflection list format.
		header = '''CrystFEL reflection list version 2.0
Symmetry: mmm
   h    k    l          I    phase   sigma(I)   nmeas'''
		footer = '''End of reflections
Generated by CrystFEL'''
		Crystfel_out_put_arry = np.hstack((out_put_arry[:,1:2],out_put_arry[:,0:1],out_put_arry[:,2:3],out_put_arry[:,13:14],np.zeros((out_put_arry.shape[0],1)),out_put_arry[:,14:15]/np.sqrt(out_put_arry[:,3:4]),out_put_arry[:,3:4]))
		ind = np.isnan(Crystfel_out_put_arry).any(axis=1)
		Crystfel_out_put_arry = Crystfel_out_put_arry[~ind,:]
		np.savetxt('all_HKL_crystfel.hkl',Crystfel_out_put_arry,header=header,footer=footer,fmt=['%4d','%4d','%4d','%10.2f','%8s','%10.2f','%7d'],comments='')
		return


	def clean(self):
		del self.arry
##################################################

class Frame:
		
	def __init__(self, dataset_obj, frame_no):
		self.arry = dataset_obj.arry[dataset_obj.arry[:,0]==int(frame_no),:]
		self.frame_no = frame_no
		reference_arry = np.genfromtxt('ethc_mk.pdb.hkl',skip_header=3,skip_footer=2,usecols=(0,1,2,3))
		self.reference_arry = np.hstack((reference_arry[:,1:2],reference_arry[:,0:1],reference_arry[:,2:]))
		self.get_reflections()
		
		self.total_inte = self.arry[:,3].sum()
		self.clean()
		
	def get_reflections(self):
		hkl_arry = np.unique(self.arry[:,4:7], axis=0)
		self.hkl_arry = hkl_arry.astype(np.int)
		self.hkl_total_no = self.hkl_arry.shape[0]
		self.reflection_obj_list = []
		for i in range(self.hkl_total_no):
			hkl = self.hkl_arry[i,:]
			reflection = Reflection(self,hkl)
			self.reflection_obj_list.append(reflection)
	
	def find_hkl(self, hkl):
		reflection_obj = None
		ind = (self.hkl_arry==np.array(hkl).astype(np.int)).all(axis=1).nonzero()[0]
		if len(ind)!=0:
			reflection_obj = self.reflection_obj_list[ind[0]]
		return reflection_obj

	@staticmethod
	def extend_HKL(HKL):
		HKL = np.array(HKL,dtype=np.int)
		H = HKL[0]
		K = HKL[1]
		L = HKL[2]
		ext_arry = np.array([[H,K,L],[-H,K,L],[H,-K,L],[H,K,-L],[H,-K,-L],[-H,K,-L],[-H,-K,L],[-H,-K,-L]])
		ext_arry = np.unique(ext_arry,axis=0)
		return ext_arry


	def find_HKL(self, HKL):
		ext_arry = Frame.extend_HKL(HKL)
		#print(ext_arry)
		reflection_obj = []
		
		for m in range(ext_arry.shape[0]):
			hkl = ext_arry[m,:]
			ind = (self.hkl_arry==np.array(hkl).astype(np.int)).all(axis=1).nonzero()[0]
			if len(ind)!=0:
				reflection_obj.append(self.reflection_obj_list[ind[0]])
		return reflection_obj


	def clean(self):
		del self.arry
	
	def adj_k_in(self):
		# add the k_in_adj attribute to pixels
		k_c0 = np.array([(1556-1594)*75e-6,(1748-1764)*75e-6,0.1291])
		k_c0 = k_c0/np.linalg.norm(k_c0)*k0

		with h5py.File('/home/lichufen/CCB_ind/scan_corrected_00135.h5','r') as f:
			#pu_arry = np.array(f['/data/data'][self.frame_no,1676:1748,1524:1556])
			pu_arry = np.array(f['/data/data'][self.frame_no,:,:])
		for r in self.reflection_obj_list:
			ind = (self.reference_arry[:,0:3]==np.abs(r.hkl)).all(axis=1).nonzero()[0]
			if len(ind)!=0:
				ind = ind[0]
				r.reference_I = self.reference_arry[ind,3]
			else:
				r.reference_I = np.nan
			
			for p in r.pixel_obj_list:
				p.k_in_adj = p.k_in
				###### compute the 
				x_ind, y_ind = (p.k_in_adj/p.k_in_adj[2]*0.1291)[0:2]
				x_ind = np.rint((x_ind/75e-6)+1594).astype(np.int)
				y_ind = np.rint((y_ind/75e-6)+1764).astype(np.int)
				pu = pu_arry[y_ind-2:y_ind+2,x_ind-2:x_ind+2].mean()/1e6
				if (pu>2):
					p.pu = pu
				else:
					p.pu = np.nan
				p.inte_adj = p.inte/p.pu

				p.diffraction_eff = p.inte/r.reference_I
			
			#r.pu_value = pu_arry[1676:1748,1524:1556].mean()/1e6
			r.pu_value = np.nanmean(np.array([p.pu for p in r.pixel_obj_list]))
		return


	def get_diff_arry(self,bins=200,thld=1e0,plot=False,save=False):
		diff_eff_arry = []
		kx_arry = []
		ky_arry = []
		for r in self.reflection_obj_list:
			for p in r.pixel_obj_list:
				if (not np.isnan(p.diffraction_eff)) and (p.diffraction_eff<thld) :
					diff_eff_arry.append(p.diffraction_eff)
					kx_arry.append(p.k_in[0])
					ky_arry.append(p.k_in[1])
		
		kx_arry = np.array(kx_arry)
		ky_arry = np.array(ky_arry)
		diff_eff_arry = np.array(diff_eff_arry)			
		bins_arry_x=np.linspace(-10e8,10e8,bins+1)
		bins_arry_y=np.linspace(-10e8,10e8,bins+1)
		bins_ind_x=np.digitize(kx_arry,bins_arry_x)	
		bins_ind_y=np.digitize(ky_arry,bins_arry_y)
		Int_arry = np.zeros((bins_arry_x.shape[0],bins_arry_y.shape[0]))
		counter = np.zeros((bins_arry_x.shape[0],bins_arry_y.shape[0]))	
		for m in range(kx_arry.shape[0]):
			Int_arry[bins_ind_x[m]-1,bins_ind_y[m]-1]+=diff_eff_arry[m]
			counter[bins_ind_x[m]-1,bins_ind_y[m]-1]+=1
		Int_arry = Int_arry/(counter+sys.float_info.epsilon)
		KX,KY=np.meshgrid(bins_arry_x,bins_arry_y)
		
		
		diff_dict = {'kx_arry':kx_arry,'ky_arry':ky_arry,'diff_eff_arry':diff_eff_arry,'bins_arry_x':bins_arry_x,'bins_arry_y':bins_arry_y,'KX':KX,'KY':KY,'Int_arry':Int_arry}
		
		if save:
			file_name_base = 'Dif_EFF_fr%d.pkl'%(self.frame_no)
			with open(file_name_base,'wb') as f:
				pk.dump(diff_dict,f)
			
		if plot:
			fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(15,4))
			im = ax[0].pcolor(KX,KY,np.log10(Int_arry.T))
			plt.colorbar(im,ax=ax[0])
			ax[0].axis('equal')
			ax[0].set_xlabel('k_in_x(A^-1)')
			ax[0].set_ylabel('k_in_y(A^-1)')
			ax[0].set_title('diffraction efficiency,log\nframe: %d'%(self.frame_no))
			im = ax[1].pcolor(KX,KY,Int_arry.T)
			im.set_clim(0,0.2)
			plt.colorbar(im,ax=ax[1])
			ax[1].axis('equal')
			ax[1].set_xlabel('k_in_x(A^-1)')
			ax[1].set_ylabel('k_in_y(A^-1)')
			ax[1].set_title('diffraction efficiency,linear\nframe: %d'%(self.frame_no))
			ax[2].hist(np.log10(diff_eff_arry),bins=200)
			ax[2].set_label('log10(diffraction efficiency)')
			ax[2].set_title('histogram')
			plt.show()


		return diff_dict
	


class Reflection:
	def __init__(self, frame_obj, hkl):
		hkl = np.array(hkl)
		self.hkl = hkl
		self.arry = frame_obj.arry[(frame_obj.arry[:,4:7]==hkl).all(axis=1),:]
		self.pixel_total_no = self.arry.shape[0]
		self.ref_inte_sum = self.arry[:,3].sum()
		self.ref_inte_ave = self.ref_inte_sum/self.pixel_total_no
		self.get_pixels()
		self.clean()

	def get_pixels(self):
		self.pixel_obj_list = []
		for i in range(self.pixel_total_no):
			pixel = Pixel(self,i)
			self.pixel_obj_list.append(pixel)

	def clean(self):
		del self.arry

	def get_inte_adj(self):
		if hasattr(self.pixel_obj_list[0], 'inte_adj'):
			#ref_inte_sum_adj = np.nansum(np.array([p.inte_adj for p in self.pixel_obj_list]))
			#ref_inte_ave_adj = np.nanmean(np.array([p.inte_adj for p in self.pixel_obj_list]))
			ref_inte_sum_adj = self.ref_inte_sum/self.pu_value
			ref_inte_ave_adj = self.ref_inte_ave/self.pu_value
						
			self.ref_inte_sum_adj = ref_inte_sum_adj
			self.ref_inte_ave_adj = ref_inte_ave_adj
		else:
			sys.exit('no attribute inte_adj for Pixel object')
		

class Pixel:
	pixel_size = 75e-6 # in m
	
	def __init__(self, reflection_obj, pixel_id):
		self.pixel_id = pixel_id
		self.arry = reflection_obj.arry[pixel_id,:]
		self.xy = self.arry[1:3]
		self.inte = self.arry[3]
		self.k_out = self.arry[7:10]
		self.k_in = self.arry[10:13]
	


if __name__=='__main__':
	K_map_file_name = os.path.abspath(sys.argv[1])
	#rank = int(sys.argv[2]) # the rank of the hkl accoridng to redundancy in ascending order
	h = int(sys.argv[2])
	k = int(sys.argv[3])
	l = int(sys.argv[4])
	dset = Dataset(K_map_file_name)
	ind = np.argsort(dset.redundancy)
	#hkl = dset.hkl_arry[ind[rank]]
	hkl = (h,k,l)
	dd = dset.search_hkl(hkl)
	#dd = dset.search_HKL(hkl)

	#for frame in dset.frame_obj_list:
		#print('frame %d has %d reflctions'%(frame.frame_no,frame.hkl_total_no))
	print('This dataset has {0:d} unique HKL measured'.format(dset.HKL_arry.shape[0]))
	#[print('HKL: ',*dset.HKL_arry[m,:],'red: ',dset.redundancy[m]) for m in range(dset.HKL_arry.shape[0])]
	#plt.figure(figsize=(5,5))
	#total_inte_arry = np.array([f.total_inte/f.hkl_total_no for f in dset.frame_obj_list])
	#plt.plot(dset.frame_arry, total_inte_arry)
	#plt.xlabel('frame')
	#plt.ylabel('total signal intensity per reflection')
	#plt.show()
	fig, ax = plt.subplots(nrows=1, ncols=2 ,figsize=(12,5))
	plt.title('hkl: (%2d,%2d,%2d) '%(hkl[0],hkl[1],hkl[2]))
	#ax[0].plot(dd['frame_ar'],dd['inte_ave_ar'],'bx-')
	ax[0].plot(dd['frame_ar'],dd['inte_ave_adj_ar'],'bx')
	#ax[1].plot(dd['frame_ar'],dd['inte_sum_ar'],'rx-')
	ax[1].plot(dd['frame_ar'],dd['inte_sum_adj_ar'],'rx')
	
	#ax[2].plot(dd['frame_ar'],dd['pixel_total_no_ar'],'kx-')
	ax[0].set_xlabel('frame')
	ax[1].set_xlabel('frame')
	#ax[2].set_xlabel('frame')
	ax[0].set_ylabel('average intensity_adj')
	ax[1].set_ylabel('sum intensity_adj')
	#ax[2].set_ylabel('# of pixels')
	plt.axis('tight')
	
	fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
	labels=['frame'+str(frame_no) for frame_no in dd['frame_ar']]
	plt.title('hkl: (%2d,%2d,%2d) '%(hkl[0],hkl[1],hkl[2]))
	ax[0].plot(dd['frame_ar'],dd['inte_ave_ar'],'bx-')
	ax[1].plot(dd['frame_ar'],dd['inte_sum_ar'],'rx-')
	ax[0].set_xlabel('frame')
	ax[1].set_xlabel('frame')
	ax[0].set_ylabel('average intensity')
	ax[1].set_ylabel('sum intensity')
	plt.axis('tight')

	fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
	labels=['frame'+str(frame_no) for frame_no in dd['frame_ar']]
	for r in dd['r_obj_list']:
		k_out_arry = np.array([]).reshape(-1,3)
		k_in_arry = np.array([]).reshape(-1,3)
		labels.append
		for p in r.pixel_obj_list:
			k_out_arry = np.vstack((k_out_arry ,p.k_out.reshape(1,3)))
			k_in_arry = np.vstack((k_in_arry, p.k_in_adj.reshape(1,3)))
		ax[0].scatter(k_out_arry[:,0],k_out_arry[:,1],s=1,marker='x')
		ax[1].scatter(k_in_arry[:,0],k_in_arry[:,1],s=5,marker='x')
	ax[0].set_title('k_out scatter for hkl: (%2d, %2d, %2d)'%(hkl[0],hkl[1],hkl[2]))
	ax[1].set_title('k_in scatter for hkl: (%2d, %2d, %2d)'%(hkl[0],hkl[1],hkl[2]))
	ax[0].axis('equal')
	ax[1].axis('equal')
	ax[0].legend(labels)
	ax[1].legend(labels)
	plt.show()
