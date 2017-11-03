import numpy as np
import scipy.misc as sp
import glob
from math import floor, ceil
import os
import random
class cityscape_images_reader():
	def shuffle_data(self):
		self.indices=np.arange(self.size)
		np.random.shuffle(self.indices)
		
		self.paths=list(np.array(self.paths)[self.indices])
		#self.labels=list(np.array(self.labels)[self.indices])

	def __init__(self,data_dir,batch_size,consider_all_images=False,image_size=None):
		self.data_dir=data_dir
		self.batch_size=batch_size
		self.consider_all_images=consider_all_images
		self.cities=[]
		self.path_to_cities=[]
		self.paths=[] # this list will contain total filepath of images being considered
		if(self.consider_all_images):
			self.directories=['train','test','val']
		else:
			self.directories=['train']
		for dir in self.directories:
			k=next(os.walk(os.path.join(self.data_dir,'leftImg8bit',dir)))[1]
			self.cities.append(k)
			for city in k:
				self.path_to_cities.append(os.path.join(self.data_dir,'leftImg8bit',dir,city))
		for path_to_a_city in self.path_to_cities:
			k=[x[2] for x in os.walk(path_to_a_city)]
			for image in k[0]:
				self.paths.append(os.path.join(path_to_a_city,image))
		if image_size is None:
			self.image_size=sp.imread(self.paths[0]).shape
		else:
			self.image_size=image_size
		self.size=np.size(self.paths)
		self.n_batches=ceil(self.size*1.0/self.batch_size)
		self.reset_reader()
		self.shuffle_data()
	def reset_reader(self):
		self.cursor=0
		self.epoch=0
		self.batch_num=0

	def next_batch(self):
		if(self.batch_num<self.n_batches-1):
			self.chunk_images=self.paths[self.cursor:self.cursor+self.batch_size]
			#self.chunk_labels=self.labels[self.cursor:self.cursor+self.batch_size]
			self.cursor=(self.cursor+self.batch_size)%self.size;


		elif(self.batch_num==self.n_batches-1):
			residue=self.size-self.cursor;
			self.chunk_images=np.concatenate([self.paths[-1*residue:],self.paths[:self.batch_size-residue]])
			#self.chunk_labels=np.concatenate([self.labels[-1*residue:],self.labels[:self.batch_size-residue]])
			self.cursor=0
			# self.shuffle_data();
			# self.epoch=self.epoch+1;
			# self.batch_num=0;
		self.batch_num=self.batch_num+1
		data=[sp.imresize(sp.imread(image_path, mode='RGB'),[self.image_size[0],self.image_size[1]]) for image_path in self.chunk_images]
		#lab=self.chunk_labels
		return [np.stack(data)]#,np.stack(lab)]
#cityscape_images_reader('/vulcan/scratch/koutilya/cityscapes/',100,False,image_size=[144,144,3])


class single_reader():
	def shuffle_data(self):
		self.indices=np.arange(self.size);
		np.random.shuffle(self.indices);
		self.data_files=self.data_files[self.indices];

	def __init__(self,data_dir,batch_size,image_size=None):
		self.data_dir=data_dir;
		self.batch_size=batch_size;
		self.data_files=fnmatch.filter(os.listdir(data_dir),'*.png')
		if(image_size==None):
			self.image_size=sp.imread(self.data_files[0]).shape
		else:
			self.image_size=image_size;
		self.size=np.size(self.data_files);
		self.n_batches=ceil(self.size*1.0/self.batch_size);

		self.reset_reader();
		self.shuffle_data();
	def reset_reader(self):
		self.cursor=0;
		self.epoch=0;
		self.batch_num=0;

	def next_batch(self):
		if(self.batch_num<self.n_batches-1):
			self.chunk_data=self.data_files[self.cursor:self.cursor+self.batch_size];
			self.cursor=(self.cursor+self.batch_size)%self.size;


		elif(self.batch_num==self.n_batches-1):
			residue=self.size-self.cursor;
			self.chunk_data=np.concatenate([self.data_files[-1*residue:],self.data_files[:self.batch_size-residue]])
			self.cursor=0;
			self.shuffle_data();
			# self.epoch=self.epoch+1;
			# self.batch_num=0;
		self.batch_num=self.batch_num+1

		data=[sp.imresize(sp.imread(os.path.join(data_dir,i), mode='RGB'),[self.image_size[0],self.image_size[1]],interp='bicubic') for i in self.chunk_data];
		return np.stack(data);

