from __future__ import division, print_function, absolute_import, unicode_literals
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import pandas as pd

# get the image
def load_img(path):
	img = tf.io.read_file(path)
	#assumed image is in png extension
	img = tf.image.decode_png(img)
	img = tf.image.convert_image_dtype(img, tf.float32)

	return img

def load_data():
	imgs = []
	#specify location in os.walk
	for (path,dirs,files) in os.walk("./data/satelite"):
		for file in files:
			if file[-3:] == "png":
				imgs.append(load_img(os.path.join(path,file)))
	return imgs

def preprocess_input(inputs):
	#concatenate 8 images.
	inputs = tf.cast(inputs, dtype = tf.float32)

	input_size = inputs.shape[0]

	assert input_size > 8 , "num of input images should be at least 8"

	#there are 8 images in one hour
	concat_num = 8
	indices = np.arange(concat_num) + np.arange(0,input_size,8)[:,np.newaxis]

	if input_size % 8 != 0:
		indices = indices[:-1]		
	indices = indices[...,np.newaxis]
	res = tf.gather_nd(inputs, indices)
	res = tf.reshape(res, tf.concat([res.shape[0:1],[-1],res.shape[3:]],0))
	return res

# build a model
imgs = load_data()
imgs = preprocess_input(imgs)
optimizer = tf.keras.optimizers.Adam(1e-4)

#batch_size doesnt matter. revise it.
def load_label():
	data = pd.read_excel(io='./data/labels/data_past_time.xls')
	data = data.iloc[:,1:3].as_matrix()[1:].astype(np.float32)
	data = tf.cast(data, tf.float32)
	return data

#build model
class customized_model(tf.keras.Model):
	def __init__(self):
		super(customized_model, self).__init__()
		self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
		self.seq = tf.keras.Sequential([layers.Flatten(),
										layers.Dense(5000),
										layers.Dense(2)])
	def call(self, inputs):
		x = self.vgg(inputs)
		x = self.seq(x)
		return x

model = customized_model()


#preprocess input
x = tf.keras.applications.vgg19.preprocess_input(imgs*255)
x = tf.image.resize(x, (224, 224))


labels = load_label()
#temporaily slice the labels so that it matches the size of the images
labels = labels[:imgs.shape[0]]

def objective(labels, res):
	return tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true = labels, y_pred = res))

@tf.function()
def train():
	with tf.GradientTape() as gt:
		res = model(x)
		obj = objective(labels, res)
		grads = gt.gradient(obj, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

epochs = 100
#add batch feature
for i in range(epochs):
	train()
	res = model(x)
	obj = objective(labels, res)
	print("%2.5f" % (obj))