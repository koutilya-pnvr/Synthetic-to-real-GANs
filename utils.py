import tensorflow as tf
import numpy as np
from math import ceil
import scipy.misc as sp
import math
import h5py
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import gen_nn_ops
# @ops.RegisterGradient("MaxPoolWithArgmax")
# def _MaxPoolWithArgmaxGrad(op, grad, some_other_arg):
# 	return gen_nn_ops._max_pool_grad(op.inputs[0],op.outputs[0],grad,op.get_attr("ksize"),op.get_attr("strides"),padding=op.get_attr("padding"),data_format='NHWC')

class Param_loader():
	def __init__(self,weights_path=None):
		if weights_path is not None:
			self.pretrained=True;
			self.weights_path=weights_path;
			self.file_type=self.weights_path.split('.')[-1];

			if(self.file_type=='npy'):
				self.transposer=[0,1,2,3]
				temp=np.load(weights_path)[()];
				self.layer_names=[];
				for i in temp:
					self.layer_names.append(i[0]);
				self.weight_data=dict()
				for pj in self.layer_names:
					# self.weight_data[pj,'0']=temp[pj]['weights']
					# self.weight_data[pj,'1']=temp[pj]['biases']
					self.weight_data[pj,'0']=temp[pj,'0']
					self.weight_data[pj,'1']=temp[pj,'1']




			elif(self.file_type=='h5'):
				self.transposer=[2,3,1,0]
				self.weight_data=dict();
				f=h5py.File(weights_path,'r');
				self.layer_names=[];
				for i in list(f['data']):
					if(len(f['data'][i])>0):
						self.layer_names.append(i);
				for pj in self.layer_names:
					self.weight_data[pj,'0']=f['data'][pj]['0'][:];
					self.weight_data[pj,'1']=f['data'][pj]['1'][:];

				f.close()
		else:
			self.layer_names=[]
			self.pretrained=False;

def get_deconv_filter(f_shape):
	width = f_shape[0]
	heigh = f_shape[0]
	f = ceil(width/2.0)
	c = (2 * f - 1 - f % 2) / (2.0 * f)
	bilinear = np.zeros([f_shape[0], f_shape[1]])
	for x in range(width):
		for y in range(heigh):
			value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
			bilinear[x, y] = value
	weights = np.zeros(f_shape)
	for i in range(f_shape[2]):
		weights[:, :, i, i] = bilinear
	return weights


def max_pool(x,k_shape,strides,name,padding='VALID'):
	with tf.variable_scope(name):
		pooled=tf.nn.max_pool(x,ksize=[1,k_shape[0],k_shape[1],1],strides=[1,strides[0],strides[1],1],padding=padding)
	return pooled;

def lrn(x, radius, alpha, beta, name, bias=1.0):
	return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,beta = beta, bias = bias, name = name)

def dropout(x, keep_prob,phase_train):
	return tf.nn.dropout(x, keep_prob) if phase_train is True else x

def print_shape(obj):
	print obj.name,obj.get_shape().as_list()


def conv_cond_concat(tensor,ys):
	#batch_size=tf.shape(tensor)[0]
	shape=get_shape(tensor)[1:-1]
	ysr=tf.tile(tf.reshape(ys,[-1,1,1,get_shape(ys)[-1]]),[1,shape[0],shape[-1],1])
	return tf.concat([tensor,ysr],axis=3)

def conv2d(inputT, k_shape, out_channels,name,is_training,strides=[1,1,1,1],reuse=False,padding='SAME',params=Param_loader(),trainable=False,stddev=1e-2,batch_norm=True):
	# Trainable parameter controls if the variable found in weights file should be further trained.
	in_channels = inputT.get_shape().as_list()[-1]
	with tf.variable_scope(name,reuse=reuse):
		weights_shape=[k_shape[0],k_shape[1],in_channels,out_channels]
		bias_shape=[out_channels]

		if(params.pretrained==False):
			weights=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=weights_shape,name='weights')
			biases=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=bias_shape,name='biases')

		elif(name in params.layer_names):
			weights=tf.get_variable(trainable=trainable,initializer=tf.constant_initializer(np.transpose(params.weight_data[layer_name,'0'],params.transposer)),shape=weights_shape,name='weights')
			biases=tf.get_variable(trainable=trainable,initializer=tf.constant_initializer(params.weight_data[layer_name,'1']),shape=bias_shape,name='biases')
		else:
			weights=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=weights_shape,name='weights')
			biases=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=bias_shape,name='biases')
		conv_out=tf.nn.conv2d(inputT,weights,strides=strides,padding=padding);
		out = tf.nn.bias_add(conv_out, biases)
		if(batch_norm==True):
			return batch_normalization(out,is_training,reuse=reuse)
		else:
			return out
def get_shape(tensor):
	return tensor.get_shape().as_list()
def conv2d_transpose(inputT,k_shape,out_shape,name,is_training,reuse=False,strides=[1,1,1,1],padding='SAME',params=Param_loader(),trainable=False,stddev=1e-2,batch_norm=True):
	in_channels = inputT.get_shape().as_list()[-1]
	batch_size=tf.shape(inputT)[0]
	out_channels=out_shape[-1]
	with tf.variable_scope(name,reuse=reuse):
		weights_shape=[k_shape[0],k_shape[1],out_channels,in_channels]
		bias_shape=[out_channels]

		if(params.pretrained==False):
			weights=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=weights_shape,name='weights')
			biases=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=bias_shape,name='biases')

		elif(name in params.layer_names):
			weights=tf.get_variable(trainable=trainable,initializer=tf.constant_initializer(np.transpose(params.weight_data[layer_name,'0'],params.transposer)),shape=weights_shape,name='weights')
			biases=tf.get_variable(trainable=trainable,initializer=tf.constant_initializer(params.weight_data[layer_name,'1']),shape=bias_shape,name='biases')
		else:
			weights=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=weights_shape,name='weights')
			biases=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=bias_shape,name='biases')

		

		conv_out=tf.nn.conv2d_transpose(inputT,weights,[batch_size]+out_shape,strides=strides,padding=padding)
		# out = tf.nn.bias_add(c1, biases)
		if(batch_norm==True):
			return batch_normalization(tf.reshape(conv_out,[-1]+out_shape),is_training,reuse=reuse)
		else:
			return tf.reshape(conv_out,[-1]+out_shape)
def l2_loss(tensor1,tensor2=None):
	if(tensor2==None):
		return tf.reduce_mean(tensor1*tensor1)
	else:
		return tf.reduce_mean((tensor1-tensor2)*(tensor1-tensor2))

def Activation(inputT,activation='relu'):
	
	if(activation=='none'):
		activ_function=lambda x: x
	elif(activation=='relu'):
		activ_function=lambda x: tf.nn.relu(x)
	elif(activation=='lrelu'):
		activ_function=lambda x: LeakyRelu(x,0.2)
	elif(activation=='prelu'):
		activ_function=lambda x: Parametric_Relu(x,name)
	elif(activation=='sigmoid'):
		activ_function=lambda x: tf.nn.sigmoid(x)
	elif(activation=='tanh'):
		activ_function=lambda x: tf.tanh(x)
	return activ_function(inputT)

def fc_flatten(x,num_out,name,is_training,reuse=False,activation='none',params=Param_loader(),trainable=False,stddev=1e-2,batch_norm=False):
	#Fully connected layer
	with tf.variable_scope(name,reuse=reuse):
		shape=np.array(x.get_shape().as_list())
		num_in=np.prod(shape[1:])
		x_reshaped=tf.reshape(x,[-1,num_in])  #reshape just to be safe
		# weights=get_weights_fc(params,var_name='weights',layer_name=name,shape=[num_in,num_out],trainable=trainable)
		# biases=get_biases(params,var_name='biases',shape=[num_out],layer_name=name,trainable=trainable)
		weights_shape=[num_in,num_out]
		bias_shape=[num_out]
		if(params.pretrained==False):
			weights=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=weights_shape,name='weights')
			biases=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=bias_shape,name='biases')

		elif(name in params.layer_names):
			weights=tf.get_variable(trainable=trainable,initializer=tf.constant_initializer(params.weight_data[layer_name,'0']),shape=weights_shape,name='weights')
			biases=tf.get_variable(trainable=trainable,initializer=tf.constant_initializer(params.weight_data[layer_name,'1']),shape=bias_shape,name='biases')
		else:
			weights=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=weights_shape,name='weights')
			biases=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),shape=bias_shape,name='biases')

		added=tf.nn.bias_add(tf.matmul(x_reshaped,weights),biases)
		if(batch_norm==True):
			return batch_normalization(added,is_training,reuse=reuse)
		else:
			return added		
		
def fc_convol(x,k_shape,num_out,name,reuse=False,activation='none',params=Param_loader(),trainable=False):
	with tf.variable_scope(name,reuse=reuse):
		#Fully connected layer as a convolution
		num_in=x.get_shape().as_list()[-1];	
		weights=get_weights(params,[k_shape[0],k_shape[1],num_in,num_out],var_name='fc_weights',layer_name=name,trainable=trainable)
		bias=get_biases(params,[num_out],var_name='fc_biases',layer_name=name,trainable=trainable)
		c1=tf.nn.conv2d(x,weights,strides=[1,1,1,1],padding='SAME');
		bias=tf.nn.bias_add(c1,bias);
		return Activation(bias,activation)


def batch_normalization(inputT,is_training,reuse,affine=False):
	with tf.variable_scope('batch_norm',reuse=reuse):
		if(affine==True):
			center=True;scale=True
		else:
			center=False;scale=False
		# return tf.cond(is_training,
		# 	lambda:tf.contrib.layers.batch_norm(inputT,reuse=reuse,is_training=True,updates_collections=None,center=center,scale=scale),
		# 	lambda:tf.contrib.layers.batch_norm(inputT,reuse=reuse,is_training=False,updates_collections=None,center=center,scale=scale))
		if(is_training==True):
			return tf.layers.batch_normalization(inputT,training=True,center=center,scale=scale)
		else:
			return tf.layers.batch_normalization(inputT,training=False,center=center,scale=scale)

def batch_norm_layer(inputT, is_training, scope,params,reuse,trainable):
	this_name=scope+'_bn';
	print this_name
	if(params.pretrained==False):
		return tf.contrib.layers.batch_norm(inputT,reuse=reuse,center=False,is_training=is_training,updates_collections=None, scope=this_name)
	else:
		if(this_name in params.layer_names):
			print 'pretrained BN param', this_name, 'with shape', params.weight_data[this_name,'0'].shape;
			param_initializers=dict()
			param_initializers['moving_mean']=tf.constant_initializer(params.weight_data[this_name,'0'].reshape([-1]));
			param_initializers['moving_variance']=tf.constant_initializer(params.weight_data[this_name,'1'].reshape([-1]));
			return tf.contrib.layers.batch_norm(inputT, center=True,scale=True, param_initializers=param_initializers,reuse=reuse,trainable=trainable,is_training=is_training, updates_collections=None, scope=this_name) 
		else:
			return tf.contrib.layers.batch_norm(inputT, reuse=reuse,is_training=is_training,center=True,scale=True, updates_collections=None, scope=this_name) 


def LeakyRelu(x,param):
	pos_part=tf.nn.relu(x)
	neg_part=param*(x-tf.abs(x))*0.5
	return pos_part+neg_part

def Parametric_Relu(x,name):
	with tf.variable_scope(name+'_alpha'):
		alpha=tf.get_variable(initializer=tf.constant_initializer(0.0))
	pos_part=tf.nn.relu(x)
	neg_part=alpha*(x-tf.abs(x))*0.5
	return pos_part+neg_part

def upsample(x,k_shape=[2,2],name='upsample',out_shape=None):
	in_shape=tf.get_shape(x)
	if out_shape is None:
		out_shape=[-1]+[j*i for i,j in zip(in_shape[1:-1],k_shape)]+[-1]
	shape=tf.stack([ins[0],out_shape[1],out_shape[2],in_shape[-1]])
	x_indices=tf.where(tf.equal(x,x))
	new_indices=2*x_indices[:,1:-1]
	new_indices=tf.concat([tf.reshape(x_indices[:,0],[-1,1]),new_indices,tf.reshape(x_indices[:,-1],[-1,1])],1)
	updates=tf.reshape(x,[-1])
	output=tf.scatter_nd(new_indices,updates,tf.cast(shape,tf.int64))
	return output

def downsample(x,k_shape=[2,2],name='downsample',out_shape=None):
	return x[:,0::k_shape[0],0::k_shape[1],:]

def upsample_with_pool_mask(updates, mask, ksize=[1, 2, 2, 1],out_shape=None,name=None):
	input_shape=tf.shape(updates)
	# input_shape = updates.get_shape().as_list()
	if out_shape is None:
		out_shape = (ins[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
	one_like_mask = tf.ones_like(mask)
	batch_range = tf.reshape(tf.range(out_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
	b = one_like_mask * batch_range
	y = mask // (out_shape[2] * out_shape[3])
	x = mask % (out_shape[2] * out_shape[3]) // out_shape[3]
	feature_range = tf.range(out_shape[3], dtype=tf.int64)
	f = one_like_mask * feature_range
	updates_size = tf.size(updates)
	indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
	values = tf.reshape(updates, [updates_size])
	ret = tf.scatter_nd(indices, values, out_shape)
	return ret


def xavier_initializer(kernel_size,num_filters):
	stddev = math.sqrt(2. / (kernel_size**2 * num_filters))
	return tf.truncated_normal_initializer(stddev=stddev)





