import tensorflow as tf
import sys
sys.path.insert(0, '/cfarhomes/sriram12/semantic/')
import numpy as np
import os
from image_reader import *
from cityscape_reader import *
from utils import *
from argparse import ArgumentParser
import h5py
from flip_gradient import flip_gradient
import matplotlib.pyplot as plt

class face_gen():
	def __init__(self,keep_prob=0.5,sample_size=200,weights_path=None,pretrained=False):
		self.keep_prob = keep_prob
		self.sample_size=sample_size
		self.l=1
		self.pretrained=pretrained
		self.params=Param_loader()
		if weights_path is not None:
			self.pretrained=True
			self.params=Param_loader(weights_path)

	def sample(self,z,reuse,is_training):
		gen_imgs=self.netG(z,reuse,is_training)
		return gen_imgs

	def train_gta(self,x,z,reuse,is_training=True,gen_lr=2e-4,disc_lr=2e-4):
		ys=tf.one_hot(labels,self.num_classes) 

		gen_in=conv_cond_concat(x,z)

		#features=self.netF(x,reuse=reuse,is_training=is_training)
		gen_out=self.netG(gen_in,reuse=reuse,is_training=is_training)
		[disc_out_gen_type,disc_out_gen_class]=self.netD(gen_out,ys,reuse=reuse,is_training=is_training)
		[disc_out_images_type,disc_out_images_class]=self.netD(x,ys,reuse=True,is_training=is_training)

		batch_size=tf.shape(x)[0]	

		disc_real_typeloss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.ones([batch_size],dtype=tf.int32),2),logits=disc_out_images_type))
		disc_fake_typeloss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.zeros([batch_size],dtype=tf.int32),2),logits=disc_out_gen_type))
		disc_real_classloss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=disc_out_images_class))
		disc_fake_classloss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=disc_out_gen_class))
		gen_typeloss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.ones([batch_size],dtype=tf.int32),2),logits=disc_out_gen_type))
		gen_classloss=disc_fake_classloss
		
		self.disc_loss=(disc_real_typeloss+disc_fake_typeloss+disc_real_classloss+disc_fake_classloss)
		self.gen_loss=(gen_typeloss+gen_classloss)
		
		trainable_vars=tf.trainable_variables()
		self.disc_vars=[var for var in trainable_vars if 'discriminator' in var.name]
		self.gen_vars=[var for var in trainable_vars if 'generator' in var.name]

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  		with tf.control_dependencies(update_ops):
			self.gen_trainop=tf.train.AdamOptimizer(gen_lr).minimize(self.gen_loss,var_list=self.gen_vars)
			self.disc_trainop=tf.train.AdamOptimizer(disc_lr).minimize(self.disc_loss,var_list=self.disc_vars)
		# return [disc_out_images,disc_out_generated]

	def get_shape(self,x):
		return x.get_shape().as_list();

	def netF(self,imgs_batch,reuse,is_training):
		
		# - tf.constant([129.1863,104.7624,93.5940])
		#self.inp_tensor=(self.inp_tensor-127.0)/127.0
		imgs_batch=(imgs_batch-127.0)/127.0

		# Layer 1
		conv1_1=conv2d(imgs_batch,[3,3],64,'conv1_1',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse,batch_norm=False) # D! what happens if we incorporate batchnorm here?
		conv1_1=Activation(conv1_1,'relu')
		print_shape(conv1_1)

		conv1_2=conv2d(conv1_1,[3,3],64,'conv1_2',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse)
		conv1_2=Activation(conv1_2,'relu')
		print_shape(conv1_2)
		
		# Layer 2
		conv2_1=conv2d(conv1_2,[3,3],128,'conv2_1',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse)
		conv2_1=Activation(conv2_1,'relu')
		print_shape(conv2_1)

		conv2_2=conv2d(conv2_1,[3,3],128,'conv2_2',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse)
		conv2_2=Activation(conv2_2,'relu')
		print_shape(conv2_2)
		
		# Layer 3
		conv3_1=conv2d(conv2_2,[3,3],256,'conv3_1',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse)
		conv3_1=Activation(conv3_1,'relu')
		print_shape(conv3_1)

		# conv3_2=conv2d(conv3_1,[3,3],256,'conv3_2',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse)
		# conv3_2=Activation(conv3_2,'relu')
		# print_shape(conv3_2)
		return conv3_1


	def netG(self,rand_input,reuse,is_training):
		rand_input=tf.reshape(rand_input,[-1,self.sample_size])
		gen_input=rand_input
		with tf.variable_scope('generator'):
			print_shape(gen_input)

			z_proj=fc_flatten(gen_input, np.prod([4,4,8]),'fc1',is_training=is_training,reuse=reuse)
			z_proj=Activation(z_proj,'relu')
			z_img=tf.reshape(z_proj,[-1,9,9,8])
			print_shape(z_img)
			ups1=conv2d_transpose(z_img, [3,3],[9,9,32], 'deconv1',is_training=is_training,strides=[1,2,2,1],reuse=reuse)
			ups1=Activation(ups1,'relu')
			print_shape(ups1)
			ups2=conv2d_transpose(ups1, [3,3],[18,18,64], 'deconv2',is_training=is_training,strides=[1,2,2,1],reuse=reuse)
			ups2=Activation(ups2,'relu')
			print_shape(ups2)
			ups3=conv2d_transpose(ups2, [3,3],[36,36,128], 'deconv3',is_training=is_training,strides=[1,2,2,1],reuse=reuse)
			ups3=Activation(ups3,'relu')
			print_shape(ups3)
			ups4=conv2d_transpose(ups3, [3,3],[72,72,256], 'deconv4',is_training=is_training,strides=[1,2,2,1],reuse=reuse)
			ups4=Activation(ups4,'relu')
			print_shape(ups4)
			ups5=conv2d_transpose(ups4, [3,3],[144,144,512], 'deconv5',is_training=is_training,strides=[1,2,2,1],reuse=reuse,batch_norm=False)
			ups5=conv2d(ups5,[3,3],3,'conv',is_training,params=self.params,reuse=reuse,batch_norm=False)
			ups5=Activation(ups5,'tanh')
			print_shape(ups5)
		return ups5
		
		

	def netD(self,disc_input,reuse,is_training):
		with tf.variable_scope('discriminator'):
			print_shape(disc_input)
			# Layer 1
			conv1_1=conv2d(disc_input,[3,3],64,'conv1_1',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse,batch_norm=False)
			conv1_1=Activation(conv1_1,'relu')
			conv1_2=conv2d(conv1_1,[3,3],128,'conv1_2',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse)
			conv1_2=Activation(conv1_2,'relu')
			conv2_1=conv2d(pool1,[3,3],256,'conv2_1',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse)
			conv2_1=Activation(conv2_1,'relu')
			conv2_2=conv2d(conv2_1,[3,3],256,'conv2_2',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse)
			conv2_2=Activation(conv2_2,'relu')
			conv3_1=conv2d(pool2,[3,3],512,'conv3_1',is_training,strides=[1,2,2,1],params=self.params,reuse=reuse)
			conv3_1=Activation(conv3_1,'relu')

			fc1=fc_flatten(conv3_1,4096,'fc1',is_training,params=self.params,reuse=reuse)
			fc1=dropout(fc1,self.keep_prob,is_training)
			fc2=fc_flatten(fc1,3090,'fc2',is_training,params=self.params,reuse=reuse)

			fc3=fc_flatten(conv3_1,2,'fc3',is_training,params=self.params,reuse=reuse)

			return [fc3,fc1]
		
	def netC(self,lp_input,reuse,is_training):
		fc6=fc_flatten(source_features,4096,'fc6_labels',is_training,params=self.params,reuse=reuse,relu=True)
		print_shape(fc6)
		fc6=dropout(fc6,self.keep_prob,self.is_training)
		fc7=fc_flatten(fc6,4096,'fc7_labels',phase_train=self.is_training,params=self.params,reuse=reuse,relu=True)
		print_shape(fc7)
		fc7=dropout(fc7,self.keep_prob,self.is_training)
		fc8=fc_flatten(fc7,self.num_classes,'fc8_labels',phase_train=self.is_training,params=self.params,reuse=reuse)
		print_shape(fc8)
		return fc8


def train_scene_gen():
	n_epochs=2000
	validate_every=5
	viz_every=3
	save_every=10
	base_lr=5e-6
	logfile_path='/fs/vulcan-scratch/koutilya/projects/Synthetic-to-real-GANs/models/logfile_domain'
	# model_file_path='/scratch0/sriram/models/vgg_face.h5'
	model_save_path='/fs/vulcan-scratch/koutilya/projects/Synthetic-to-real-GANs/models/cityscape_gen/cityscape_gen'
	#epoch_number=20
	source_data_dir=os.path.join('/vulcan/scratch/koutilya/cityscapes/')
	# target_data_dir=os.path.join('/scratch0/sriram/frames_aligned/')
	batch_size=50
	batch_size_valid=10
	sample_size=200

	reader=cityscape_images_reader('/vulcan/scratch/koutilya/cityscapes/',100,False,image_size=[144,144,3])
	#reader_target=face_reader(target_data_dir,batch_size,)
	# reader=face_reader(source_data_dir,batch_size,'umdfaces_train.txt',image_size=[144,144,3])
	# reader_valid=face_reader(source_data_dir,batch_size_valid,'umdfaces_val.txt',image_size=[224,224,3])
	# reader_target=face_reader(target_data_dir,batch_size,'umdvideos_train.txt',image_size=[224,224,3])
	# reader_target_valid=face_reader(target_data_dir,batch_size_valid,'umdvideos_val.txt',image_size=[224,224,3])

	image_size=reader.image_size	
	f_train=open(logfile_path,'w+')
	sess=tf.Session()
	x=tf.placeholder(tf.float32,shape=[None,image_size[0],image_size[1],image_size[2]])
	random_vec=tf.placeholder(tf.float32,shape=[None,sample_size])
	label=tf.placeholder(tf.int64, shape=[None]) 
	# target_data=tf.placeholder(tf.float32,shape=[None,image_size[0],image_size[1],image_size[2]])
	learning_rate=tf.placeholder(tf.float32,shape=[])
	# is_training=tf.placeholder(tf.bool,shape=[])
	net=face_gen(sample_size=sample_size)
	# input_features=tf.concat([source_data,target_data],axis=0) if mode=='dann' else source_data
	[disc_real_logits,disc_fake_logits,gen_imgs]=net.train(x,random_vec,reuse=False,is_training=True)

	print 'built network'

	# label_prob=tf.nn.softmax(label_logits)
	# domain_prob=tf.nn.softmax(domain_logits)
	# label_pred=tf.argmax(label_prob,axis=1)
	# domain_pred=tf.argmax(domain_prob,axis=1)
	# label_accuracy=tf.reduce_sum(tf.cast(tf.equal(label_pred,label),tf.float32))
	# domain_accuracy=tf.reduce_sum(tf.cast(tf.equal(domain_pred,domain_label),tf.float32))
	
	saver=tf.train.Saver(tf.trainable_variables())
	sess.run(tf.global_variables_initializer())
	#saver.restore(sess,model_save_path+'-'+str(epoch_number))
	print 'initialized vars'
	print tf.trainable_variables()
	#reader.epoch=epoch_number
	while(reader.epoch<n_epochs):
			cnt=0;agg_l=0;agg_d=0;
			


			while(reader.batch_num<reader.n_batches):
				cnt+=1
				[source_imgs,source_label]=reader.next_batch()
				feed_dict_train={x:source_imgs,label:source_label,learning_rate:base_lr,random_vec:np.random.standard_normal([batch_size,sample_size])}
				[dl,gl,_1,_2]=sess.run([net.disc_loss,net.gen_loss,net.gen_trainop,net.disc_trainop],feed_dict=feed_dict_train)
				print 'Training','epoch',reader.epoch,'batch',reader.batch_num,'gen_loss:',gl,'disc_loss',dl

			if((reader.epoch+1)%save_every==0):
				saver.save(sess,model_save_path,global_step=(reader.epoch+1))
			if((reader.epoch+1)%viz_every==0):
				gen_imgs=sess.run(net.sample(np.random.standard_normal([batch_size,sample_size]).astype('float32'),reuse=True,is_training=False))
				gen_imgs=gen_imgs*127+127
				viz_batch(gen_imgs,reader.epoch+1)
			reader.epoch=reader.epoch+1
			reader.batch_num=0


def viz_batch(batch,iteration):
	n=batch.shape[0]
	for i in range(n):
		plt.subplot(n/10,10,i+1)
		plt.imshow(batch[i,:].reshape([144,144,3]))	
	plt.savefig('images_'+str(iteration)+'.jpg')


if __name__=="__main__":
	parser = ArgumentParser()
	parser.add_argument('-ngpu',type=int,default=0)
	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES']=str(args.ngpu)
	train_scene_gen()
	#test_fn()
  
  
